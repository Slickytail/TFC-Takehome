"""Data loading for TinyChronos.

Data layout: data/<split>/<subset>/train-XXXXX-of-XXXXX.parquet (files keep
the train- naming in both splits; the split is the directory), GluonTS-style
— one row = one multivariate time series. Relevant columns:
  target:                list<list<float>>, shape (n_var_target, T)
  past_feat_dynamic_real: list<list<float>>, shape (n_var_past, T) — may be
                          empty (n_var_past = 0). T always matches `target`.

Parquet's minimum read unit is a row group, and in this corpus row groups are
huge (rows are whole series, so groups can be 100-400MB). True random access to
a single series is impossible without decompressing its whole row group, so we
accept a bounded amount of correlation instead:

  - At init, read parquet footers only (no data IO) to index every
    (file, row_group).
  - The in-memory buffer (buffer_bytes, split across DataLoader workers) is
    organized into buckets by total n_var. Row groups are preloaded into the
    bucket matching their n_var; rows are consumed one at a time from their
    payload, so payloads stay resident until exhausted and the buffer drains
    and refills with fresh row groups.
  - Each batch draws one bucket (probability proportional to its number of
    remaining rows) and takes one row from each of batch_size distinct
    payloads (also row-proportional), so batches mix row
    groups/files/subsets as much as possible while never mixing n_var.
  - Each row yields one randomly positioned window: sequence length and
    cutoff are drawn per batch (uniform over the configured ranges, bounded
    by the shortest row in the batch), starts are per row.
  - DataLoader workers shard the group list (i::N) and each keeps its own
    buffer sized buffer_bytes / num_workers.
"""

import glob
import json
import os
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info
from tqdm.auto import tqdm


class TinyChronosDataConfig:
    def __init__(
        self,
        root: str = "data",
        split: str = "train",  # subdirectory of root: train / test
        subsets: Optional[list[str]] = None,  # None = all subdirectories
        min_seq_len: int = 64,
        max_seq_len: int = 512,
        cutoff_low: float = 0.1,
        cutoff_high: float = 0.9,
        buffer_bytes: int = 4 * 1024**3,
        seed: int = 0,
        outlier_threshold: float = 7.0,  # |asinh-normalized| above this = missing
        max_batch_vars: int = 25,  # buckets above this n_var use batch_size 1
    ):
        self.root = root
        self.split = split
        self.subsets = subsets
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.buffer_bytes = buffer_bytes
        self.seed = seed
        self.outlier_threshold = outlier_threshold
        self.max_batch_vars = max_batch_vars

    @classmethod
    def from_dict(cls, d: dict) -> "TinyChronosDataConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "TinyChronosDataConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _list_list_to_numpy(la: pa.Array) -> list[np.ndarray]:
    """Convert a list<list<float>> arrow array (one row) to a list of (T,)
    float32 arrays, one per variable. Nulls become NaN."""
    inner = la.values
    if isinstance(inner, pa.ChunkedArray):
        inner = inner.combine_chunks()
    io = inner.offsets.to_numpy(zero_copy_only=False)
    vals = np.asarray(inner.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return [vals[io[i] : io[i + 1]].copy() for i in range(len(inner))]


def _read_series_column(table: pa.Table, column: str) -> list[list[np.ndarray]]:
    """Column of list<list<float>> -> list (per row) of lists (per var) of
    (T,) float32 arrays."""
    col = table.column(column)
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    return [_list_list_to_numpy(col[r]) for r in range(len(col))]


class TinyChronosDataset(IterableDataset):
    def __init__(self, config: TinyChronosDataConfig, batch_size: Optional[int] = None):
        """batch_size: None yields un-batched sample dicts; an int N yields
        collated batches of N samples that all share the same total n_var
        (bucketed batches; see module docstring)."""
        super().__init__()
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be None or >= 1")
        self.batch_size = batch_size
        self.config = config
        self.epoch = 0

        # Metadata-only index: (path, row_group_idx). Reading footers is cheap
        # and involves no data IO.
        self.index: list[tuple[str, int]] = []
        self.n_rows = 0
        split_dir = os.path.join(config.root, config.split)
        for subset in config.subsets or sorted(os.listdir(split_dir)):
            subset_dir = os.path.join(split_dir, subset)
            if not os.path.isdir(subset_dir):
                continue
            for path in sorted(glob.glob(os.path.join(subset_dir, "*.parquet"))):
                pf = pq.ParquetFile(path)
                self.index.extend((path, rg) for rg in range(pf.num_row_groups))
                self.n_rows += pf.metadata.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read_row_group(self, path: str, rg: int) -> list[dict]:
        """Decode one row group into per-row payloads:
        x: (n_var_target + n_var_past, T) float32, n_var_target: int,
        freq: pandas-style frequency string (used for seasonality in eval)."""
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(
            rg, columns=["target", "past_feat_dynamic_real", "freq"]
        )
        freqs = table.column("freq").to_pylist()
        targets = _read_series_column(table, "target")
        covs = _read_series_column(table, "past_feat_dynamic_real")

        rows = []
        for tgt, cov, freq in zip(targets, covs, freqs):
            if not tgt or tgt[0].shape[0] == 0:
                continue
            if tgt[0].shape[0] < self.config.min_seq_len:
                continue  # too short for any window; keeps batch length bounds valid
            x = np.stack(tgt + cov) if cov else np.stack(tgt)
            rows.append({"x": x, "n_var_target": len(tgt), "freq": freq})
        return rows

    def _build_batch(self, rows: list[dict], rng: np.random.Generator) -> dict:
        """Collate rows that all share the same total n_var into one batch.

        Sequence length and cutoff are drawn once per batch so the time axis
        stacks without padding (still uniform over the configured ranges);
        the window start is drawn per row. n_var_target may differ per row.
        """
        cfg = self.config
        B = len(rows)
        n_var = rows[0]["x"].shape[0]
        assert all(r["x"].shape[0] == n_var for r in rows)
        T_min = min(r["x"].shape[1] for r in rows)

        seq_len = int(rng.integers(cfg.min_seq_len, min(cfg.max_seq_len, T_min) + 1))
        lo = max(1, int(cfg.cutoff_low * seq_len))
        hi = min(seq_len - 1, int(cfg.cutoff_high * seq_len))
        cutoff = int(rng.integers(lo, hi + 1))  # history length
        n_future = seq_len - cutoff

        x_past = np.empty((B, n_var, cutoff), dtype=np.float32)
        x_future = np.empty((B, n_var, n_future), dtype=np.float32)
        mask_past = np.empty((B, n_var, cutoff), dtype=np.int8)
        mask_future = np.empty((B, n_var, n_future), dtype=np.int8)
        for b, row in enumerate(rows):
            T = row["x"].shape[1]
            start = int(rng.integers(0, T - seq_len + 1))
            window = row["x"][:, start : start + seq_len]
            finite = np.isfinite(window)
            x_past[b] = np.nan_to_num(window[:, :cutoff])
            x_future[b] = np.nan_to_num(window[:, cutoff:])
            mask_past[b] = finite[:, :cutoff]
            mask_future[b] = finite[:, cutoff:]

        return {
            "x_past": torch.from_numpy(x_past),
            "x_future": torch.from_numpy(x_future),
            "t_past": torch.arange(-cutoff, 0, dtype=torch.int64).expand(B, -1),
            "t_future": torch.arange(0, n_future, dtype=torch.int64).expand(B, -1),
            "mask_past": torch.from_numpy(mask_past),
            "mask_future": torch.from_numpy(mask_future),
            "n_var_target": torch.tensor(
                [r["n_var_target"] for r in rows], dtype=torch.int64
            ),
            "freq": [r["freq"] for r in rows],
        }

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 7919 * self.epoch + worker_id)

        groups = list(self.index)
        rng.shuffle(groups)
        groups = groups[worker_id::num_workers]
        group_iter = iter(groups)

        budget = max(1, cfg.buffer_bytes // num_workers)
        # Buffer organized into buckets by total n_var. Each bucket holds
        # payloads (row groups); rows are consumed one at a time so a payload
        # stays resident until exhausted and the buffer drains/refills.
        buckets: dict[int, list[tuple[list[dict], int]]] = {}  # n_var -> [(rows, nbytes)]
        bucket_rows: dict[int, int] = {}  # remaining rows per bucket
        buffered_bytes = 0
        initial_fill = True  # only the startup preload gets a progress bar

        def refill() -> None:
            nonlocal buffered_bytes, initial_fill
            show_pbar = initial_fill
            initial_fill = False
            desc = "preload" if num_workers == 1 else f"preload [w{worker_id}]"
            pbar = tqdm(
                total=budget,
                desc=desc,
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
                leave=False,
                disable=not show_pbar,
            )
            while buffered_bytes < budget:
                try:
                    path, rg = next(group_iter)
                except StopIteration:
                    break
                payload = self._read_row_group(path, rg)
                if not payload:
                    continue
                nbytes = sum(r["x"].nbytes for r in payload)
                nvar = payload[0]["x"].shape[0]
                buckets.setdefault(nvar, []).append((payload, nbytes))
                bucket_rows[nvar] = bucket_rows.get(nvar, 0) + len(payload)
                buffered_bytes += nbytes
                pbar.update(nbytes)
            pbar.close()

        def build_batch(nvar: int) -> dict:
            """Build one batch from the given bucket: one random row from each
            of batch_size distinct payloads (weighted by remaining row count),
            so a batch mixes row groups/files/subsets as much as possible.
            Buckets with very high n_var use batch size 1: memory and compute
            per row scale with n_var, so a full batch there would dominate
            step time and risk swapping."""
            nonlocal buffered_bytes
            payloads = buckets[nvar]
            k = min(self.batch_size or 1, len(payloads))
            if nvar > cfg.max_batch_vars:
                k = 1
            weights = np.array([len(r) for r, _ in payloads], dtype=np.float64)
            idx = rng.choice(len(payloads), size=k, replace=False, p=weights / weights.sum())
            rows, freqs, nvts = [], [], []
            for i in sorted(idx):  # sorted so payload removal stays valid
                rows_list, nb = payloads[i]
                j = int(rng.integers(0, len(rows_list)))
                row = rows_list.pop(j)
                row_nb = row["x"].nbytes
                bucket_rows[nvar] -= 1
                buffered_bytes -= row_nb
                if rows_list:
                    payloads[i] = (rows_list, nb - row_nb)
                else:
                    payloads[i] = ([], 0)
                rows.append(row)
                freqs.append(row["freq"])
                nvts.append(row["n_var_target"])
            # drop exhausted payloads
            buckets[nvar] = [p for p in payloads if p[0]]
            if not buckets[nvar]:
                del buckets[nvar]
                bucket_rows.pop(nvar)

            batch = self._build_batch(rows, rng)
            if self.batch_size is None:
                # unbatched mode: decompose the (B=1) batch into a sample dict
                batch = {
                    "x_past": batch["x_past"][0],
                    "x_future": batch["x_future"][0],
                    "t_past": batch["t_past"][0],
                    "t_future": batch["t_future"][0],
                    "mask_past": batch["mask_past"][0],
                    "mask_future": batch["mask_future"][0],
                    "n_var_target": nvts[0],
                    "freq": freqs[0],
                }
            else:
                batch["freq"] = freqs
            return batch

        while True:
            refill()
            live = {n: r for n, r in bucket_rows.items() if r > 0 and n in buckets}
            if not live:
                break
            # Pick a bucket with probability proportional to its remaining rows.
            nvars = list(live.keys())
            sizes = np.array([live[n] for n in nvars], dtype=np.float64)
            nvar = int(nvars[rng.choice(len(nvars), p=sizes / sizes.sum())])
            yield build_batch(nvar)
