"""Does the Transformer's uncertainty deficit come from over-parametrisation?

Table 4 shows the Transformer's predictive spread barely orders its own errors on
target-weighted samples (Spearman rho = 0.17) while the GPs reach 0.44-0.47, and
within the dropout family rho falls as capacity rises: DNN 509k parameters ->
0.251, Transformer 803k -> 0.174, matched DNN 808k -> 0.178. That is what
over-parametrisation would predict, but the Deep GP is comparably large
(731k) and still reaches 0.437, so capacity cannot be the whole story.

This is the controlled test. Several Transformer capacities are trained on ONE
fixed labelled set with ONE fixed train/val split, and each is scored with the
same metrics and the same MC-dropout machinery the paper uses. Holding the data
fixed is the point: a capacity sweep run as separate AL loops would confound
capacity with the different acquisition trajectories the loops would follow.

Prediction under the hypothesis: rho rises as capacity falls, while RMSE
worsens. If instead rho stays flat near 0.17 while RMSE degrades, capacity is
exonerated and the deficit belongs to the covariance structure, which is what a
last-layer Laplace or GP head would address.

Usage:
    python scripts/transformer_capacity_sweep.py \\
        --run-dir /ptmp/jwuerzin/output/active_learning_transformer_entropy_batch_cold_seed1_20260803_180047 \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (tag, d_model, nhead, num_layers, dim_feedforward); the last reproduces the
# production model exactly (803,330 parameters) and is the control.
LADDER = [
    ("tiny",       32, 4, 2,  64),
    ("small",      64, 4, 2, 128),
    ("medium",     64, 4, 3, 256),
    ("production", 128, 4, 3, 512),
]


@click.command()
@click.option("--run-dir", required=True,
              help="AL run whose final labelled set and split are reused.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True)
@click.option("--eval-size", default=20_000, show_default=True,
              help="Eval rows per set, matching evaluate_uq.py's default.")
@click.option("--mc-samples", default=30, show_default=True)
@click.option("--epochs", default=10_000, show_default=True)
@click.option("--patience", default=200, show_default=True)
@click.option("--dropout", default=0.1, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", default=1, show_default=True)
def main(run_dir, output_dir, mcmc_data_dir, eval_size, mc_samples, epochs,
         patience, dropout, device, seed):
    import torch
    import multiprocessing as mp
    from analyse_runs import load_run
    from pmssm.data import load_mcmc_data, transform_y
    from pmssm.models.transformer import PMSSMTransformerTabular
    from pmssm.training import train_model_worker
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "euq", str(_REPO_ROOT / "scripts" / "evaluate_uq.py"))
    euq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(euq)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "capacity_sweep_work"
    work.mkdir(exist_ok=True)

    # ---- one fixed labelled set and split, shared by every capacity ---------
    run = load_run(run_dir)
    # load_run may hand back numpy or torch depending on how the state was
    # written; normalise once rather than assuming.
    X = torch.as_tensor(np.asarray(run.X)).float()
    Y = torch.as_tensor(np.asarray(run.Y)).float().ravel()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_val = max(1, int(0.2 * len(X)))
    idx_val, idx_train = perm[:n_val], perm[n_val:]
    click.echo(f"[cap] labelled set {len(X)} rows from {Path(run_dir).name}")
    click.echo(f"[cap] fixed split: {len(idx_train)} train / {len(idx_val)} val")

    # ---- eval set: the target-weighted reference, as in Table 4 -------------
    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, max_samples=500_000)
    Xm = torch.as_tensor(np.asarray(Xm.numpy() if hasattr(Xm, "numpy") else Xm)).float()
    Ym = torch.as_tensor(np.asarray(Ym.numpy() if hasattr(Ym, "numpy") else Ym)).float().ravel()
    sel = torch.randperm(len(Xm), generator=torch.Generator().manual_seed(123))[:eval_size]
    X_ev, Y_ev = Xm[sel], Ym[sel]
    y_true = transform_y(Y_ev, target="DMRD").numpy().astype(np.float64)
    click.echo(f"[cap] eval set {len(X_ev)} rows (target-weighted reference)")

    results = {}
    for tag, dm, nh, nl, ff in LADDER:
        n_par = sum(p.numel() for p in PMSSMTransformerTabular(
            d_model=dm, nhead=nh, num_layers=nl,
            dim_feedforward=ff, dropout=dropout).parameters())
        ck = work / f"tf_{tag}.pt"
        click.echo(f"\n[cap] {tag}: d_model={dm} layers={nl} ff={ff} "
                   f"({n_par:,} parameters)")
        torch.manual_seed(seed)
        np.random.seed(seed)
        q = mp.Queue()
        train_model_worker(device, X, Y.unsqueeze(-1), idx_train, idx_val,
                           epochs, dropout, q, f"cap_{tag}", str(work),
                           str(work), ck, None, True, patience, "log", "DMRD",
                           None, arch="transformer",
                           tf_d_model=dm, tf_nhead=nh, tf_num_layers=nl,
                           tf_dim_feedforward=ff)
        res = q.get()
        model = PMSSMTransformerTabular(d_model=dm, nhead=nh, num_layers=nl,
                                        dim_feedforward=ff, dropout=dropout)
        model.load_state_dict(torch.load(ck, map_location=device))
        pred = euq._predict_dropout(model, X_ev, X[idx_train], Y[idx_train],
                                    mc_samples, device)
        m = euq._uq_metrics(y_true, pred)
        results[tag] = {"params": int(n_par), "d_model": dm, "num_layers": nl,
                        "dim_feedforward": ff,
                        "val_loss": float(res.get("best_val_loss", float("nan"))),
                        **{k: m[k] for k in ("rmse", "mean_sigma", "var_z",
                                             "spearman_sigma_abserr", "ause",
                                             "miscalibration_area", "nlpd",
                                             "crps")}}
        click.echo(f"[cap] {tag:<11} params={n_par:>8,}  rmse={m['rmse']:.3f}  "
                   f"var_z={m['var_z']:7.2f}  rho={m['spearman_sigma_abserr']:+.3f}  "
                   f"ause={m['ause']:.3f}")

    p = out_dir / "transformer_capacity_sweep.json"
    p.write_text(json.dumps({"config": {"run_dir": run_dir, "eval_size": eval_size,
                                        "mc_samples": mc_samples, "seed": seed},
                             "results": results}, indent=1))
    click.echo(f"\n[cap] wrote {p}")
    click.echo("\n  params      rmse    var_z      rho    ause")
    for tag, r in sorted(results.items(), key=lambda kv: kv[1]["params"]):
        click.echo(f"  {r['params']:>8,}  {r['rmse']:6.3f} {r['var_z']:8.2f} "
                   f"{r['spearman_sigma_abserr']:+7.3f} {r['ause']:7.3f}   {tag}")


if __name__ == "__main__":
    main()
