"""TinyChronos: a multivariate time series transformer.

Inputs to the model:
  observations: (B, n_var, t) floats. Historical values, plus arbitrary
    (e.g. zero) placeholders for future timesteps the model should predict.
  time_ids: (B, t) integers. Negative for observed past timesteps, with the
    first unknown/future timestep having time id 0 (and increasing from there).
  masks: (B, n_var, t) integers. 1 where the value is observed, 0 where it is
    missing/unknown.

The caller is responsible for padding: to get forecasts, append zeroed
observations with mask=0 and non-negative time ids. The model returns
quantile predictions for *every* input timestep, shape (B, n_var, t, q);
the caller selects the future positions it cares about.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import load_file, save_file


class TinyChronosConfig:
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        quantiles: Optional[list[float]] = None,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.quantiles = (
            list(quantiles) if quantiles is not None else [round(0.1 * i, 1) for i in range(1, 10)]
        )
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)

    @classmethod
    def from_dict(cls, d: dict) -> "TinyChronosConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "TinyChronosConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """F.scaled_dot_product_attention, preferring the flash backend.

    On CPU the default backend materializes the (…, t, t) score matrix, which
    autograd retains for backward — with n_var=45, t=512 that's ~750MB per
    layer and blows up memory. The flash CPU kernel never materializes it.
    """
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    except RuntimeError:
        return F.scaled_dot_product_attention(q, k, v)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings along the sequence dim.

    q, k: (..., n_heads, t, head_dim); positions: (..., t) numeric. The cos/sin
    tensors get a singleton head dim so they broadcast over attention heads.
    """
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )  # (head_dim/2,)
    angles = positions.unsqueeze(-1).float() * inv_freq  # (N, t, head_dim/2)
    emb = torch.cat([angles, angles], dim=-1)  # (..., t, head_dim)
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    cos, sin = cos.unsqueeze(-3), sin.unsqueeze(-3)  # (..., 1, t, head_dim)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class TimeAttention(nn.Module):
    """Full self-attention across the time axis; each variable attends only
    to itself. Uses RoPE with the time ids and no causal masking."""

    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, n_var, t, h); time_ids: (B, n_var, t)
        B, n_var, t, h = x.shape
        qkv = self.qkv(x).view(B, n_var, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)  # (B, n_var, H, t, dh)

        # Flash attention requires 4-D q/k/v: merge batch and variable axes.
        q, k, v = (
            x.reshape(B * n_var, self.n_heads, t, self.head_dim) for x in (q, k, v)
        )
        q, k = apply_rope(q, k, time_ids.reshape(B * n_var, t), self.rope_theta)

        out = _sdpa(q, k, v)  # (B*n_var, H, t, dh)
        out = (
            out.view(B, n_var, self.n_heads, t, self.head_dim)
            .transpose(2, 3)
            .reshape(B, n_var, t, h)
        )
        return self.proj(out)


class GroupAttention(nn.Module):
    """Self-attention across the variable axis; each timestep attends only to
    itself, i.e. attention between the n_var values at the same timestep.
    No position encoding."""

    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_var, t, h)
        B, n_var, t, h = x.shape
        qkv = self.qkv(x).view(B, n_var, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 2, 4, 1, 5).unbind(0)  # (B, t, H, n_var, dh)

        # Flash attention requires 4-D q/k/v: merge batch and time axes.
        q, k, v = (
            x.reshape(B * t, self.n_heads, n_var, self.head_dim) for x in (q, k, v)
        )

        out = _sdpa(q, k, v)  # (B*t, H, n_var, dh)
        out = (
            out.view(B, t, self.n_heads, n_var, self.head_dim)
            .transpose(2, 3)
            .reshape(B, n_var, t, h)
        )
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TinyChronosLayer(nn.Module):
    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        self.time_norm = nn.LayerNorm(config.d_model)
        self.time_attn = TimeAttention(config)
        self.group_norm = nn.LayerNorm(config.d_model)
        self.group_attn = GroupAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, n_var, t, h); time_ids: (B, n_var, t)
        x = x + self.time_attn(self.time_norm(x), time_ids)
        x = x + self.group_attn(self.group_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyChronosEmbedding(nn.Module):
    """Embeds the per-token features [observation, normalized time id, mask]
    (dim 3) into the model dimension via 3 -> 4h -> h with a ReLU."""

    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 4 * config.d_model),
            nn.ReLU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyChronosQuantileHead(nn.Module):
    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.n_quantiles),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class TinyChronos(nn.Module):
    def __init__(self, config: TinyChronosConfig):
        super().__init__()
        self.config = config
        self.embedding = TinyChronosEmbedding(config)
        self.layers = nn.ModuleList(
            TinyChronosLayer(config) for _ in range(config.n_layers)
        )
        self.quantile_head = TinyChronosQuantileHead(config)
        # Quantile levels as a buffer: saved in / loaded from the weights so a
        # checkpoint is self-describing about what its output columns mean.
        self.register_buffer(
            "quantiles", torch.tensor(config.quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        observations: torch.Tensor,
        time_ids: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        observations: (B, n_var, t) float
        time_ids: (B, t) integer, negative for the past, 0 for first future step
        masks: (B, n_var, t) integer, 1 = observed, 0 = missing/unknown
        returns: (B, n_var, t, n_quantiles)
        """
        masks = masks.float()
        x = observations * masks  # zero out missing/unknown values

        emb_t = (time_ids / self.config.max_seq_len).float()  # (B, t)
        emb_t = emb_t.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, n_var, t)

        x = torch.stack([x, emb_t, masks], dim=-1)  # (B, n_var, t, 3)
        x = self.embedding(x)  # (B, n_var, t, h)

        # RoPE position per (variable, timestep): (B, n_var, t)
        rope_ids = time_ids.unsqueeze(1).expand(-1, x.shape[1], -1)

        for layer in self.layers:
            x = layer(x, rope_ids)

        return self.quantile_head(x)

    def save_pretrained(self, path: str) -> None:
        """Save config.json and model.safetensors into a directory."""
        os.makedirs(path, exist_ok=True)
        self.config.to_json(os.path.join(path, "config.json"))
        save_file(
            self.state_dict(),
            os.path.join(path, "model.safetensors"),
            metadata={"format": "pt"},
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "TinyChronos":
        """Load a model saved with save_pretrained."""
        config = TinyChronosConfig.from_json(os.path.join(path, "config.json"))
        model = cls(config)
        state_dict = load_file(os.path.join(path, "model.safetensors"))
        model.load_state_dict(state_dict)
        return model
