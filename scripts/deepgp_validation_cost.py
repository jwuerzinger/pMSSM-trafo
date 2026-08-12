"""Where does the Deep GP actually fail, and which operation does it?

The 20k probe aborts with a ROCm "Memory access fault by GPU", preceded by
"Device memory allocation size is too small for TRSM", at n_train = 57982 having
succeeded at 46454. This reproduces that in isolation, in minutes rather than
hours, so the cause can be pinned down instead of inferred from AL run logs.

What it has established, including what it refuted:

  * The unbatched validation pass in the Deep GP's training loop (the whole
    validation set in one call under fast_pred_var(False)) was the leading
    hypothesis. It is NOT the cause and NOT a cost problem: at benchmark scale it
    takes 0.012 s against a 3.365 s training epoch, i.e. 0.4%, because gpytorch
    takes its lazy path rather than a dense n_val^3 Cholesky. Batching it is
    2.3x SLOWER and does not prevent the fault. A FLOP count had predicted the
    opposite; the timing settled it, and the batching was reverted.
  * The wall reproduces here: n_train 40000 fine, 60000 faults, which brackets
    the real run. Peak PyTorch memory at 40000 is only 1.46 GB on a 64 GB
    device, so this is not capacity exhaustion. rocBLAS/rocSOLVER workspaces are
    allocated outside PyTorch's allocator and do not appear in that figure,
    which is consistent with the TRSM message.

Each phase runs in its own subprocess (--only), because a GPU memory fault
aborts the process rather than raising: running all phases together identifies
the size that fails but not the operation that fails.

Usage:
    python scripts/deepgp_validation_cost.py --mode time --n-train 12581
    python scripts/deepgp_validation_cost.py --mode isolate \\
        --sizes 40000,45000,50000,55000,60000
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts"),
           str(_REPO_ROOT / "al_pmssmwithgp" / "model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build(n_train: int, n_val: int, device: str, jitter: float):
    """A Deep GP on random data of the requested shape, as the driver builds it."""
    import torch
    from gp_pipeline.models.deep_gp import DeepGP
    g = torch.Generator().manual_seed(0)
    xt = torch.rand(n_train, 19, generator=g)
    yt = torch.randn(n_train, generator=g)
    xv = torch.rand(n_val, 19, generator=g)
    yv = torch.randn(n_val, generator=g)
    m = DeepGP(xt.to(device), yt.to(device), xv.to(device), yv.to(device), 19,
               lengthscale=1.0, noise=1e-2, num_inducing_max=256,
               inducing_strategy="vanilla", kernel="RBF", num_samples=8, seed=1)
    return m.to(device)


def _time_one(n_train: int, n_val: int, device: str, jitter: float,
              only: str = "all") -> dict:
    """One training epoch and one validation pass, timed separately."""
    import gpytorch
    import torch
    from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
    from torch.utils.data import DataLoader, TensorDataset

    m = _build(n_train, n_val, device, jitter)
    mll = DeepApproximateMLL(VariationalELBO(m.likelihood, m, n_train))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(m.x_train, m.y_train), batch_size=256,
                        shuffle=True)

    def sync():
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         gpytorch.settings.fast_pred_var(False):
        # ---- one training epoch (minibatched, as in the real loop) ----------
        t_train = float("nan")
        if only in ("all", "train"):
            m.train(); m.likelihood.train()
            sync(); t0 = time.perf_counter()
            for xb, yb in loader:
                opt.zero_grad()
                with gpytorch.settings.num_likelihood_samples(m.num_samples):
                    loss = -mll(m(xb.to(device)), yb.to(device))
                loss.backward(); opt.step()
            sync(); t_train = time.perf_counter() - t0

        # ---- validation exactly as the loop does it: ONE unbatched call -----
        m.eval(); m.likelihood.eval()
        t_val_unbatched = float("nan")
        if only in ("all", "val_unbatched"):
            sync(); t0 = time.perf_counter()
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(m.num_samples):
                out = m.output_layer(m.hidden_layer(m.x_valid.to(device)))
                _ = -mll(out, m.y_valid.to(device))
            sync(); t_val_unbatched = time.perf_counter() - t0

        # ---- and batched, which is the proposed fix -------------------------
        t_val_batched = float("nan")
        if only in ("all", "val_batched"):
          sync(); t0 = time.perf_counter()
          with torch.no_grad(), gpytorch.settings.num_likelihood_samples(m.num_samples):
            tot = 0.0
            for i in range(0, n_val, 1024):
                xb = m.x_valid[i:i + 1024].to(device)
                yb = m.y_valid[i:i + 1024].to(device)
                tot += float(-mll(m.output_layer(m.hidden_layer(xb)), yb))
          sync(); t_val_batched = time.perf_counter() - t0

    peak = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
    return {"n_train": n_train, "n_val": n_val,
            "train_epoch_s": t_train,
            "val_unbatched_s": t_val_unbatched,
            "val_batched_s": t_val_batched,
            "peak_gpu_gb": peak}


@click.command()
@click.option("--mode", type=click.Choice(["time", "sweep", "_child", "isolate"]),
              default="time", show_default=True)
@click.option("--only", type=click.Choice(["build", "train", "val_unbatched",
                                           "val_batched", "all"]),
              default="all", show_default=True,
              help="Run a single phase. With one phase per subprocess, an abort "
                   "identifies the operation, not merely the size.")
@click.option("--n-train", default=12581, show_default=True,
              help="Benchmark scale by default (iteration 40 of a 500-batch run).")
@click.option("--n-val", default=0, show_default=True,
              help="0 = n_train/4, the driver's 80/20 split.")
@click.option("--sizes", default="20000,40000,60000,80000,120000,160000",
              show_default=True, help="sweep mode: n_train values to try.")
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--jitter", default=1e-3, show_default=True)
@click.option("--out", default="/ptmp/jwuerzin/analysis/all_runs/deepgp_validation_cost.json",
              show_default=True)
def main(mode, only, n_train, n_val, sizes, device, jitter, out):
    nv = n_val or max(1, n_train // 4)

    if mode == "isolate":
        # Bisect the wall, and for each size run every phase in its own
        # subprocess so the aborting operation is named.
        rows = []
        for s_ in (int(x) for x in sizes.split(",")):
            line = {"n_train": s_, "n_val": max(1, s_ // 4)}
            for phase_name in ("build", "train", "val_unbatched", "val_batched"):
                cmd = [sys.executable, __file__, "--mode", "_child",
                       "--only", phase_name, "--n-train", str(s_),
                       "--device", device, "--jitter", str(jitter)]
                pr = subprocess.run(cmd, capture_output=True, text=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
                fault = "Memory access fault" in pr.stderr
                line[phase_name] = ("ok" if pr.returncode == 0
                                    else f"FAULT" if fault else f"rc={pr.returncode}")
            rows.append(line)
            click.echo(f"  n_train={line['n_train']:>7} n_val={line['n_val']:>6}  "
                       + "  ".join(f"{k}={line[k]}" for k in
                                   ("build", "train", "val_unbatched", "val_batched")))
        Path(out).write_text(json.dumps({"mode": "isolate", "rows": rows}, indent=1))
        click.echo(f"  wrote {out}")
        return

    if mode == "_child":
        # Run by the sweep in a subprocess so a GPU abort cannot take the parent
        # down with it. Result goes to stdout as one JSON line.
        print(json.dumps(_time_one(n_train, nv, device, jitter, only)), flush=True)
        return

    if mode == "time":
        r = _time_one(n_train, nv, device, jitter)
        tot = r["train_epoch_s"] + r["val_unbatched_s"]
        click.echo(f"\n  n_train={r['n_train']}  n_val={r['n_val']}")
        click.echo(f"    training epoch      {r['train_epoch_s']:8.3f} s")
        click.echo(f"    validation UNBATCHED{r['val_unbatched_s']:8.3f} s"
                   f"   ({r['val_unbatched_s']/max(tot,1e-9)*100:5.1f}% of an iteration)")
        click.echo(f"    validation batched  {r['val_batched_s']:8.3f} s"
                   f"   speedup {r['val_unbatched_s']/max(r['val_batched_s'],1e-9):.1f}x")
        click.echo(f"    peak GPU            {r['peak_gpu_gb']:8.2f} GB")
        Path(out).write_text(json.dumps({"mode": "time", "result": r}, indent=1))
        return

    # ---- sweep: find the wall, one subprocess per size ----------------------
    results = []
    for s in (int(x) for x in sizes.split(",")):
        cmd = [sys.executable, __file__, "--mode", "_child", "--n-train", str(s),
               "--device", device, "--jitter", str(jitter)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        dt = time.perf_counter() - t0
        line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
        if p.returncode == 0 and line:
            r = json.loads(line); r["ok"] = True
            click.echo(f"  n_train={s:>7} n_val={r['n_val']:>7}  OK   "
                       f"train {r['train_epoch_s']:6.2f}s  "
                       f"val_unbatched {r['val_unbatched_s']:6.2f}s  "
                       f"val_batched {r['val_batched_s']:6.2f}s  "
                       f"peak {r['peak_gpu_gb']:.2f} GB")
        else:
            fault = "Memory access fault" in p.stderr
            r = {"n_train": s, "n_val": max(1, s // 4), "ok": False,
                 "returncode": p.returncode, "gpu_fault": fault,
                 "elapsed_s": dt,
                 "stderr_tail": p.stderr.strip().splitlines()[-3:]}
            click.echo(f"  n_train={s:>7} n_val={r['n_val']:>7}  FAILED rc={p.returncode}"
                       f"{'  (GPU memory fault)' if fault else ''}")
        results.append(r)
    ok = [r["n_train"] for r in results if r.get("ok")]
    bad = [r["n_train"] for r in results if not r.get("ok")]
    click.echo(f"\n  largest n_train that worked: {max(ok) if ok else 'none'}")
    click.echo(f"  smallest n_train that failed: {min(bad) if bad else 'none'}")
    Path(out).write_text(json.dumps({"mode": "sweep", "results": results}, indent=1))
    click.echo(f"  wrote {out}")


if __name__ == "__main__":
    main()
