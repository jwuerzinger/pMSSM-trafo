"""Where does entropy-batch selection stop working, as a function of batch size?

The large-batch probes run top_k rather than entropy_batch, and the reason is
structural: ``select_entropy_batch_mc`` forms an (n_pool, n_pool) sample
covariance, and its failsafe raises n_pool to n_select whenever a caller asks
for a batch larger than the pool. So a 20k batch silently demands a
20000 x 20000 covariance, plus a dense torch.eye of the same shape, plus
whatever the iterative selector allocates on top. This measures the actual
ceiling instead of asserting one.

Each size runs in its OWN process (``--only <n>``), because the failure mode on
this platform is not always a catchable exception: an oversized allocation can
surface as a rocBLAS workspace warning followed by a GPU memory-access fault
that aborts the interpreter. A single process sweeping upward would report the
first fault as the limit and lose every larger size, and would also mask a
recoverable OOM behind a hard abort.

Usage:
    # one size, in isolation (what the sweep driver calls)
    python scripts/entropy_batch_limit.py --only 20000 --device cuda:0

    # print the sweep plan
    python scripts/entropy_batch_limit.py --sizes 500,1000,2000,5000,10000,20000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pmssm.selection import select_entropy_batch_mc                # noqa: E402

# Matches the production acquisition: T=30 MC-dropout passes over a 1e6
# candidate pool. T is what bounds the covariance rank, so it must be realistic.
T_DEFAULT = 30
N_CAND_DEFAULT = 1_000_000


def _synthetic(n_cand, T, dim=19, seed=0):
    """Predictions with the shape and rank structure the real loop produces.

    Values do not need to be physical: the cost and the failure mode depend on
    the shapes, on T bounding the covariance rank, and on enough candidates
    surviving the tolerance cut to fill the pool.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n_cand, dim, generator=g)
    # Means straddling the threshold so the tolerance cut keeps a large pool.
    mean = torch.randn(n_cand, 1, generator=g) * 0.5
    spread = torch.rand(n_cand, 1, generator=g) * 0.3 + 0.02
    preds = mean.unsqueeze(0) + spread.unsqueeze(0) * torch.randn(
        T, n_cand, 1, generator=g)
    var = preds.var(dim=0, unbiased=True)
    return X, preds, mean, var


def _run_one(n_select, device, n_cand, T, entropy_pool_size):
    """Return a record for one batch size, or raise."""
    import logging
    log = logging.getLogger("entropy-limit")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X, preds, mean, var = _synthetic(n_cand, T)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    idx = select_entropy_batch_mc(
        X, preds, mean, var, n_select=n_select,
        n_pool=entropy_pool_size, device=device, logger=log,
        tolerance_sampling=0.0, proximity_sampling=0.0)
    dt = time.time() - t0

    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device.startswith("cuda") and torch.cuda.is_available() else float("nan"))
    n_pool_eff = max(n_select, entropy_pool_size)
    return {
        "n_select": n_select,
        "n_pool_effective": n_pool_eff,
        "cov_gib_float32": n_pool_eff ** 2 * 4 / 2**30,
        "seconds": dt,
        "peak_gib": peak,
        "n_selected": int(len(idx)),
        "unique": int(len(np.unique(idx))),
        "status": "ok",
    }


@click.command()
@click.option("--only", type=int, default=None,
              help="Run exactly this batch size and print one JSON line. "
                   "Isolation is the point: see the module docstring.")
@click.option("--sizes", default="500,1000,2000,5000,10000,20000",
              show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--n-candidates", default=N_CAND_DEFAULT, show_default=True)
@click.option("--mc-samples", "T", default=T_DEFAULT, show_default=True)
@click.option("--entropy-pool-size", default=5000, show_default=True,
              help="Production default. The failsafe raises it to n_select "
                   "whenever n_select exceeds it, which is the effect under test.")
@click.option("--out", default=None, help="Append the JSON record here.")
def main(only, sizes, device, n_candidates, T, entropy_pool_size, out):
    if only is None:
        plan = [int(s) for s in sizes.split(",") if s.strip()]
        click.echo("# run each of these in its own process:")
        for n in plan:
            eff = max(n, entropy_pool_size)
            click.echo(f"#   n_select={n:>6}  n_pool_eff={eff:>6}  "
                       f"cov={eff**2*4/2**30:6.2f} GiB (float32)")
        return

    try:
        rec = _run_one(only, device, n_candidates, T, entropy_pool_size)
    except torch.OutOfMemoryError as exc:                       # noqa: PERF203
        rec = {"n_select": only, "status": "OOM", "error": str(exc)[:200]}
    except Exception as exc:                                    # noqa: BLE001
        rec = {"n_select": only, "status": type(exc).__name__, "error": str(exc)[:200]}

    line = json.dumps(rec)
    click.echo("RESULT " + line)
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
