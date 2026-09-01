"""CLI training script for TinyChronos.

Usage:
    python -m tinychronos.train \
        --model-config config/model.json \
        --data-config config/dataloader.json \
        --train-config config/train.json

Per batch (batched by total n_var; n_var_target may differ per sample):
  1. compute per-variable mean/std over the observed history only (no grad)
  2. normalize past + future with asinh((v - mu) / sigma) (chronos-2 style;
     inverse sinh keeps outliers from dominating)
  3. mark as missing anything the dataloader flagged NaN/inf OR whose
     |asinh-normalized| value exceeds the outlier threshold
  4. model input mask: past & observed & non-extreme (the future is always
     missing from the model's point of view)
  5. quantile loss per element everywhere, then masked to future targets that
     are observed and non-extreme, averaged per batch element then across the
     batch
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, fields
from itertools import islice
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tinychronos.dataloader import TinyChronosDataConfig, TinyChronosDataset
from tinychronos.evaluate import evaluate_model
from tinychronos.modeling import TinyChronos, TinyChronosConfig
from tinychronos.utils import asinh_normalize, history_stats


@dataclass
class TrainConfig:
    resume_from: Optional[str] = None  # model.safetensors file or checkpoint dir
    max_steps: int = 10_000
    lr: float = 1e-4
    save_to: str = "checkpoints"
    save_every: int = 1_000
    eval_every: int = 500
    batch_size: int = 1
    warmup_steps: int = 100  # linear warmup from 0 to lr
    eval_max_samples: int = 100  # fixed-size sample of the valid split for eval
    grad_accum_steps: int = 1  # update the optimizer every N batches
    # ~median observed global grad norm (see scripts/grad_norm_probe.py);
    # typical steps clip mildly, spikes are bounded.
    clip_grad_norm: float = 10.0  # 0 disables gradient clipping

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def quantile_loss(
    y_hat: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Pinball loss summed over quantiles.

    y_hat: (B, n_var, t, q); y: (B, n_var, t); quantiles: (q,)
    returns (B, n_var, t)
    """
    q = quantiles.view(1, 1, 1, -1)
    return (
        q * torch.relu(y.unsqueeze(-1) - y_hat)
        + (1 - q) * torch.relu(y_hat - y.unsqueeze(-1))
    ).sum(dim=-1)


def build_loss_mask(batch: dict, missing: torch.Tensor) -> torch.Tensor:
    """(B, n_var, t) mask: 1 on future target positions (t >= 0,
    d < n_var_target[b]) that are observed AND non-extreme.

    Deliberately not the inverse of the model input mask: historical NaNs have
    input mask 0 but carry no loss, while the loss cares about future targets
    that are genuinely observed. `missing` is the (B, n_var, t) bool from the
    training loop (genuine NaN/inf OR post-normalization outlier).
    """
    t_past = batch["t_past"].shape[1]
    B, n_var, t_total = missing.shape
    is_target = torch.arange(n_var) < batch["n_var_target"].unsqueeze(1)  # (B, n_var)
    loss_mask = torch.zeros(B, n_var, t_total)
    loss_mask[:, :, t_past:] = (
        ~missing[:, :, t_past:] & is_target.unsqueeze(2)
    ).float()
    return loss_mask


def save_checkpoint(
    path: str, model: TinyChronos, optimizer, scheduler, step: int
) -> None:
    """Full checkpoint: model weights + config (save_pretrained) plus trainer
    state, so training can be resumed from `path`."""
    model.save_pretrained(path)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(path, "trainer.pt"),
    )


def train(
    model_config: TinyChronosConfig,
    data_config: TinyChronosDataConfig,
    train_config: TrainConfig,
) -> None:
    model = TinyChronos(model_config)
    start_step = 0

    if train_config.resume_from is not None:
        src = train_config.resume_from
        if os.path.isdir(src):
            model = TinyChronos.from_pretrained(src)
            trainer_state = torch.load(
                os.path.join(src, "trainer.pt"), weights_only=False
            )
            start_step = trainer_state["step"]
            print(f"resumed weights + trainer state from {src} (step {start_step})")
        else:
            from safetensors.torch import load_file

            model.load_state_dict(load_file(src))
            print(f"resumed weights from {src}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    def warmup_lambda(step: int) -> float:
        if train_config.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / train_config.warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    if start_step > 0 and os.path.isdir(train_config.resume_from):
        state = torch.load(
            os.path.join(train_config.resume_from, "trainer.pt"), weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    dataset = TinyChronosDataset(data_config, batch_size=train_config.batch_size)
    dataloader = DataLoader(dataset, batch_size=None)  # dataset yields batches

    # Fixed evaluation set: the same samples every eval. All valid-split
    # subsets (the file-level 10% carve-out of the old test split; see
    # scripts/split_test_valid.py), capped at eval_max_samples. The eval
    # dataset gets a small buffer: it only needs to yield eval_max_samples
    # samples. The full test split is reserved for evaluate.py.
    eval_config = TinyChronosDataConfig.from_dict(
        {
            **data_config.to_dict(),
            "split": "valid",
            "subsets": None,
            "buffer_bytes": min(data_config.buffer_bytes, 256 * 1024**2),
        }
    )
    eval_samples = list(
        islice(TinyChronosDataset(eval_config), train_config.eval_max_samples)
    )
    if not eval_samples:
        print("warning: no eval samples found in the valid split")

    print(
        f"model: {sum(p.numel() for p in model.parameters()):,} params | "
        f"{len(dataset.index)} row groups | "
        f"lr={train_config.lr} warmup={train_config.warmup_steps}"
    )

    model.train()
    step = start_step  # counts batches, not optimizer updates
    ema_loss: Optional[float] = None
    accum = train_config.grad_accum_steps
    while step < train_config.max_steps:
        pbar = tqdm(
            total=train_config.max_steps,
            initial=step,
            desc="train",
            dynamic_ncols=True,
        )
        for batch in dataloader:
            with torch.no_grad():
                x_past = batch["x_past"]
                x_future = batch["x_future"]

                # 1-2. per-variable stats over observed history, asinh normalize
                mu, sigma = history_stats(x_past, batch["mask_past"])
                x = torch.cat([x_past, x_future], dim=2)
                x_n = asinh_normalize(x, mu, sigma)

                # 3. missing = genuine NaN/inf (dataloader mask) OR extreme values
                # (|asinh-normalized| above the threshold). Comparisons detach.
                observed = (
                    torch.cat([batch["mask_past"], batch["mask_future"]], dim=2) > 0
                )
                extreme = x_n.abs() > data_config.outlier_threshold
                missing = (~observed) | extreme  # (B, n_var, t)

                # 4. model input mask: past & observed & non-extreme; the future is
                # always missing. The model multiplies observations by the mask.
                time_ids = torch.cat([batch["t_past"], batch["t_future"]], dim=1)
                is_future = (time_ids >= 0).unsqueeze(1)  # (B, 1, t) broadcasts
                model_mask = ~missing & ~is_future

            y_hat = model(x_n, time_ids, model_mask)

            # quantile loss per element everywhere
            loss = quantile_loss(y_hat, x_n, model.quantiles)

            # loss on observed, non-extreme future targets only; average
            # per batch elem, then across the batch
            loss_mask = build_loss_mask(batch, missing)
            per_elem = (loss * loss_mask).sum(dim=(1, 2)) / loss_mask.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
            total_loss = per_elem.mean()

            # 7. backward; the optimizer only updates every `accum` batches, so
            # changing grad_accum_steps does not change total runtime (only
            # how many batch gradients each update averages)
            (total_loss / accum).backward()
            step += 1
            pbar.update(1)
            if step % accum != 0:
                continue
            if train_config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.clip_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            item = total_loss.item()
            ema_loss = item if ema_loss is None else 0.95 * ema_loss + 0.05 * item
            pbar.set_postfix(
                loss=f"{ema_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if step % train_config.eval_every == 0 and eval_samples:
                metrics = evaluate_model(model, eval_samples)
                pbar.write(
                    f"eval @ step {step}: "
                    + " ".join(f"{k} MASE={v:.4f}" for k, v in metrics.items())
                )

            if step % train_config.save_every == 0 or step >= train_config.max_steps:
                ckpt = os.path.join(train_config.save_to, f"checkpoint-{step:08d}")
                save_checkpoint(ckpt, model, optimizer, scheduler, step)
                pbar.write(f"saved {ckpt}")

            if step >= train_config.max_steps:
                break
        # Flush a partial accumulation window at the epoch boundary (slightly
        # underweighted since fewer than `accum` losses contributed).
        if step < train_config.max_steps and step % accum != 0:
            if train_config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.clip_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.close()
        dataset.set_epoch(dataset.epoch + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyChronos")
    parser.add_argument(
        "--model-config", default="config/model.json", help="path to model config json"
    )
    parser.add_argument(
        "--data-config",
        default="config/dataloader.json",
        help="path to dataloader config json",
    )
    parser.add_argument(
        "--train-config",
        default="config/train.json",
        help="path to training config json (defaults to TrainConfig defaults)",
    )
    args = parser.parse_args()

    model_config = TinyChronosConfig.from_json(args.model_config)
    data_config = TinyChronosDataConfig.from_json(args.data_config)
    train_config = (
        TrainConfig.from_json(args.train_config) if args.train_config else TrainConfig()
    )
    train(model_config, data_config, train_config)


if __name__ == "__main__":
    main()
