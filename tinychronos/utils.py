"""Shared numerical helpers for TinyChronos."""

import re

import torch

EPS = 1e-6

# Season lengths per base frequency, gluonts-style (DEFAULT_SEASONALITIES).
DEFAULT_SEASONALITIES = {
    "T": 1440,
    "S": 60,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "Q": 4,
    "A": 1,
    "Y": 1,
}


def get_seasonality(freq: str) -> int:
    """Season length (in steps) for a pandas-style offset alias, matching
    gluonts.time_feature.get_seasonality. E.g. 'H' -> 24, '15T' -> 96,
    'W-SUN' -> 1, 'Q-DEC' -> 4, '4S' -> 15."""
    match = re.match(r"(\d*)(.*)", freq)
    n = int(match.group(1) or 1)
    base = match.group(2).split("-")[0]
    # Month/quarter/year start conventions share the base period length.
    if base.endswith("S") and base[:-1] in DEFAULT_SEASONALITIES:
        base = base[:-1]
    return max(1, DEFAULT_SEASONALITIES.get(base, 1) // n)


def history_stats(
    x_past: torch.Tensor, mask_past: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-variable mean/std over observed history only. x_past, mask_past:
    (B, n_var, t_past). Returns mu, sigma: (B, n_var, 1)."""
    with torch.no_grad():
        count = mask_past.sum(dim=2, keepdim=True).clamp_min(1.0)
        mu = (x_past * mask_past).sum(dim=2, keepdim=True) / count
        var = (((x_past - mu) * mask_past) ** 2).sum(dim=2, keepdim=True) / count
        sigma = var.sqrt().clamp_min(EPS)
    return mu, sigma


def asinh_normalize(
    x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """chronos-2 style normalization; inverse sinh keeps outliers from
    dominating. Invertible: x = sinh(x_norm) * sigma + mu."""
    return torch.asinh((x - mu) / sigma)
