"""Evaluation for TinyChronos: MASE for the model against baselines.

Two entry points:
  - `evaluate_model(model, samples)`: called from the training loop on a fixed
    sample of the valid split (no grad, eval mode).
  - CLI: `python -m tinychronos.evaluate --checkpoint CKPT [--max-samples N]`
    runs the same metrics over the whole test split.

The valid split is a file-level ~10% carve-out of the test split (see
scripts/split_test_valid.py): in-training validation uses only it; the CLI
reports final numbers on the untouched test split.

Metrics: MASE (mean absolute scaled error), computed in the per-variable
linear-normalized space (mean 0, variance 1 over the observed history — not
asinh). The model's median quantile prediction is converted from asinh space
via sinh, baselines are standardized with the same statistics, and each
variable's errors are scaled by the mean absolute first difference of its
observed history (also in normalized space; the per-variable sigma cancels, so
this equals raw-space MASE up to degenerate (near-constant) variables, which
fall back to unscaled absolute error). A naive last-value baseline is computed
on the same positions for comparison.

Shared forward-pass helpers (`history_stats`, `asinh_normalize`) live here so
that train.py can import them without a circular import.

Note: the dataloader masks indicate genuine NaN/inf in both regions. The
model input hides the future (mask 0) as usual; scoring covers all observed
future target positions — no outlier filtering, so evaluation is on the raw
data (the training-time outlier_threshold is intentionally not applied
here).
"""

import argparse
from itertools import islice
from typing import Callable, Iterable, Optional

import torch
from tqdm.auto import tqdm

from tinychronos.dataloader import TinyChronosDataConfig, TinyChronosDataset
from tinychronos.modeling import TinyChronos
from tinychronos.utils import (
    EPS,
    asinh_normalize,
    get_seasonality,
    history_stats,
)  # noqa: F401 (helpers re-exported)

# ---------------------------------------------------------------------------
# Baselines. Register new ones with @baseline; they automatically join the
# reported metrics.
# ---------------------------------------------------------------------------

Baselines = dict[str, Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]]
BASELINES: Baselines = {}


def baseline(fn: Callable) -> Callable:
    """Register a forecast baseline fn(x_past, mask_past, n_future,
    seasonality) -> (n_var, n_future) predictions in the raw data space."""
    BASELINES[fn.__name__] = fn
    return fn


@baseline
def naive(
    x_past: torch.Tensor,
    mask_past: torch.Tensor,
    n_future: int,
    seasonality: int = 1,
) -> torch.Tensor:
    """Last observed value per variable, held constant."""
    t = torch.arange(1, x_past.shape[1] + 1)
    idx = (mask_past * t).argmax(dim=1)  # index of the last observed value
    last = x_past[torch.arange(x_past.shape[0]), idx]
    last = last * mask_past.any(dim=1)  # 0 if a variable has no observations
    return last.unsqueeze(1).expand(-1, n_future)


@baseline
def seasonal_naive(
    x_past: torch.Tensor,
    mask_past: torch.Tensor,
    n_future: int,
    seasonality: int = 1,
) -> torch.Tensor:
    """Repeat the last observed season: prediction at future step j is the
    observed value one full season before, wrapping within the final season
    of the history. Positions whose source value was unobserved fall back to
    the last observed value (like `naive`). Falls back to `naive` when the
    history is shorter than one season."""
    t_past = x_past.shape[1]
    m = seasonality if t_past > seasonality else 1  # too short for a season
    idx = t_past - m + (torch.arange(n_future) % m)
    out = x_past[:, idx]
    out_observed = mask_past[:, idx].bool()

    # Fallback value: last observed value per variable.
    t = torch.arange(1, t_past + 1)
    last_idx = (mask_past * t).argmax(dim=1)
    last = x_past[torch.arange(x_past.shape[0]), last_idx]
    last = last * mask_past.any(dim=1)

    return torch.where(out_observed, out, last.unsqueeze(1).expand(-1, n_future))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _median_index(quantiles: torch.Tensor) -> int:
    return int((quantiles - 0.5).abs().argmin().item())


def _scaled_errors(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    x_past: torch.Tensor,
    mask_past: torch.Tensor,
    n_var_target: int,
    seasonality: int,
) -> torch.Tensor:
    """Elementwise |error| / per-variable seasonal-naive scale, on the future
    target region. All tensors are in the same (linear-normalized) space.
    y_pred, y_true: (n_var, n_future); x_past, mask_past: (n_var, t_past).
    The scale is the mean absolute m-step difference of the observed history
    (the in-sample MAE of the seasonal naive forecast), matching gluonts.
    Returns (n_var_target, n_future)."""
    t_past = x_past.shape[1]
    m = seasonality if t_past > seasonality else 1  # not enough history
    obs = mask_past[:, m:] * mask_past[:, :-m]
    scale = ((x_past[:, m:] - x_past[:, :-m]).abs() * obs).sum(dim=1) / obs.sum(
        dim=1
    ).clamp_min(1.0)
    # Degenerate (near-constant) histories: fall back to unscaled absolute
    # error instead of a tiny epsilon, which would blow up the MASE.
    scale = torch.where(scale < EPS, torch.ones_like(scale), scale)
    err = (y_pred - y_true).abs()[:n_var_target] / scale[:n_var_target].unsqueeze(1)
    return err


def evaluate_model(
    model: TinyChronos,
    samples: Iterable[dict],
    baselines: Optional[Baselines] = None,
) -> dict[str, float]:
    """MASE for the model (median quantile) and each baseline over `samples`.

    Unlike the training loss, no outlier filtering is applied: every future
    target position observed in the raw data (dataloader mask, i.e. genuine
    NaN/inf excluded) is scored.

    Follows the GIFT-eval / gluonts convention: the scale is the in-sample
    MAE of the seasonal naive forecast (seasonality from the sample's `freq`
    string), and per-series MASEs are averaged. Errors are computed in the
    per-variable linear-normalized space (mean 0, variance 1 over observed
    history). samples: iterable of un-batched dataloader sample dicts;
    baselines forecast in the raw space and are standardized here.
    Returns {"model": mase, "<baseline name>": mase, ...}.
    """
    if baselines is None:
        baselines = BASELINES
    names = ["model", *baselines]
    totals = {name: 0.0 for name in names}
    count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="eval", dynamic_ncols=True):
            x_past, x_future = sample["x_past"], sample["x_future"]
            mask_past = sample["mask_past"]
            n_var_target = sample["n_var_target"]
            n_future = x_future.shape[1]

            # Model forecast: median quantile, sinh maps asinh space ->
            # linear-normalized space (mean 0, var 1).
            mu, sigma = history_stats(x_past.unsqueeze(0), mask_past.unsqueeze(0))
            x_n = asinh_normalize(
                torch.cat([x_past, x_future], dim=1).unsqueeze(0), mu, sigma
            )
            masks = torch.cat(
                [mask_past, torch.zeros_like(sample["mask_future"])], dim=1
            ).unsqueeze(0)
            time_ids = torch.cat([sample["t_past"], sample["t_future"]]).unsqueeze(0)
            y_hat = model(x_n, time_ids, masks)
            median = y_hat[0][:, -n_future:, _median_index(model.quantiles)]
            pred = torch.sinh(median)
            y_true = (x_future - mu[0]) / sigma[0]
            x_past_lin = (x_past - mu[0]) / sigma[0]
            seasonality = get_seasonality(sample["freq"])

            # Score all observed future targets (raw data, no outlier filter).
            valid = sample["mask_future"][:n_var_target] > 0
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            errs = {
                "model": _scaled_errors(
                    pred, y_true, x_past_lin, mask_past, n_var_target, seasonality
                )
            }
            for name, fn in baselines.items():
                base_pred = fn(x_past, mask_past, n_future, seasonality)
                base_pred_lin = (base_pred - mu[0]) / sigma[0]
                errs[name] = _scaled_errors(
                    base_pred_lin,
                    y_true,
                    x_past_lin,
                    mask_past,
                    n_var_target,
                    seasonality,
                )

            # Per-series MASE (over valid positions), averaged across samples
            # (gluonts convention).
            for name in names:
                totals[name] += (errs[name] * valid).sum().item() / n_valid
            count += 1

    if was_training:
        model.train()
    if count == 0:
        return {name: float("nan") for name in names}
    return {name: totals[name] / count for name in names}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TinyChronos checkpoint")
    parser.add_argument("--checkpoint", required=True, help="save_pretrained dir")
    parser.add_argument(
        "--data-config",
        default="config/dataloader.json",
        help="path to data config json",
    )
    parser.add_argument("--split", default="test", help="which split to evaluate")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="limit number of samples"
    )
    args = parser.parse_args()

    model = TinyChronos.from_pretrained(args.checkpoint)
    data_config = TinyChronosDataConfig.from_json(args.data_config)
    data_config.split = args.split
    dataset = TinyChronosDataset(data_config)

    samples: Iterable[dict] = dataset
    if args.max_samples is not None:
        samples = islice(dataset, args.max_samples)

    metrics = evaluate_model(model, samples)
    print(f"checkpoint: {args.checkpoint} | split: {args.split}")
    for name, mase in metrics.items():
        print(f"  {name}: MASE = {mase:.4f}")


if __name__ == "__main__":
    main()
