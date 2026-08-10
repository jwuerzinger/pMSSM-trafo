"""Omega h^2 vs a gaugino/higgsino mass parameter, colored by LSP type.

Reads the random-pool ntuples (standard validity filter), classifies each
point's LSP from the mixing-matrix fractions (``pmssm.visualization.
classify_lsp_type``: dominant of bino N11^2 / wino N12^2 / higgsino
N13^2+N14^2, "mixed" below ``LSP_PURITY_MIN``), and renders two panels
restricted to |param| < cut: linear Omega h^2 and log(Omega h^2 / 0.12).

The default (M_2, cut 300 GeV) exposes the wino modality: the chargino
coannihilation collapse with the three funnel prongs at small |M_2|, the
higgsino/mixed band climbing toward the target with mass, and the bino
overabundant scatter above.

Usage:
    python scripts/plot_omega_vs_m2_lsp.py
    python scripts/plot_omega_vs_m2_lsp.py --param M_1 --cut 150
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pmssm.data import LSP_FRAC_BRANCHES  # noqa: E402
from pmssm.visualization import (  # noqa: E402
    LSP_PURITY_MIN,
    LSP_TYPE_COLORS,
    LSP_TYPE_NAMES,
    classify_lsp_type,
)


@click.command()
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--param", default="M_2", show_default=True,
              type=click.Choice(["M_1", "M_2", "mu"]),
              help="Mass parameter on the x axis (ntuple branch IN_<param>).")
@click.option("--cut", default=300.0, show_default=True,
              help="|param| window (GeV).")
@click.option("--true-value", default=0.12, show_default=True)
def main(baseline_data_dir, output_dir, param, cut, true_value):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branch = f"IN_{param}"
    om_l, x_l, fr_l = [], [], []
    for fn in sorted(glob.glob(f"{baseline_data_dir}/ntuple.*.root")):
        import uproot
        t = uproot.open(fn)["susy"]
        o = t["MO_Omega"].array(library="np")
        mh = t["SP_m_h"].array(library="np")
        x = t[branch].array(library="np")
        fr = np.stack([t[b].array(library="np") for b in LSP_FRAC_BRANCHES],
                      axis=1).astype(np.float64)
        # The ntupler leaves the fraction branches at -1 when the LSP is not a
        # neutralino. Such rows have no bino/wino/higgsino composition at all,
        # but a raw -1 passes the isfinite check inside classify_lsp_type and
        # lands in "mixed" (max < LSP_PURITY_MIN). Coerce to NaN so they are
        # dropped, matching pmssm.data._load_lsp_fracs. Without this the
        # "mixed" class is dominated by sneutrino/stau LSPs, which are 86% of
        # in-band pool rows, rather than by well-tempered neutralinos.
        fr[(fr < 0).any(axis=1)] = np.nan
        m = (o > 0) & (o < 1) & (mh != -1) & (np.abs(x) < cut)
        om_l.append(o[m])
        x_l.append(x[m])
        fr_l.append(fr[m])
    om = np.concatenate(om_l)
    x = np.concatenate(x_l)
    cls = classify_lsp_type(np.concatenate(fr_l))
    click.echo(f"[lsp] n={len(om):,} inside |{param}|<{cut:.0f} "
               f"(purity threshold {LSP_PURITY_MIN}): "
               + ", ".join(f"{LSP_TYPE_NAMES[k]}={int((cls == k).sum()):,}"
                           for k in range(4)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for k in (0, 1, 2, 3):
        m = cls == k
        if not m.any():
            continue
        axes[0].scatter(x[m], om[m], s=2, alpha=0.4, color=LSP_TYPE_COLORS[k],
                        label=f"{LSP_TYPE_NAMES[k]} (n={int(m.sum())})")
        axes[1].scatter(x[m], np.log(om[m] / true_value), s=2, alpha=0.4,
                        color=LSP_TYPE_COLORS[k], label=LSP_TYPE_NAMES[k])
    axes[0].axhline(true_value, color="black", ls="--", lw=1,
                    label=rf"$\Omega h^2={true_value}$")
    axes[0].axhspan(true_value * 0.9, true_value * 1.1, color="gray", alpha=0.15)
    axes[0].set_ylim(-0.03, 1.02)
    axes[0].set_ylabel(r"$\Omega h^2$")
    axes[0].legend(fontsize=9, markerscale=3)
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].axhspan(np.log(0.9), np.log(1.1), color="gray", alpha=0.15)
    axes[1].set_ylabel(rf"log($\Omega h^2$/{true_value})")
    axes[1].legend(fontsize=9, markerscale=3, loc="lower left")
    for ax in axes:
        ax.set_xlim(-cut, cut)
        ax.set_xlabel(param)
    fig.tight_layout()
    out = Path(output_dir) / f"omega_vs_{param}_lsp_type.png"
    fig.savefig(out, dpi=150)
    click.echo(f"[lsp] wrote {out}")


if __name__ == "__main__":
    main()
