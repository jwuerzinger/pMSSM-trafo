#!/usr/bin/env python
"""Build a figure-only companion document for a non-default AL target.

Mirrors the paper draft's figure inventory for a variant sweep (by default the
ExpR / SModelS exclusion-boundary branch) and emits a LaTeX document containing
the figures and nothing else: no prose, no interpretation. Captions name the
quantity and the script that produced it so each panel is traceable.

Every figure in the paper draft gets a row in the inventory table, marked
available or not-yet-available with the reason, so what is missing is explicit
rather than silently absent.

Re-run after more of the sweep drains; it only emits figures that exist.

    ./.pixi/envs/rocm/bin/python scripts/build_expr_figure_report.py \
        --fig-dir /ptmp/jwuerzin/analysis/expr_runs \
        --out /viper/u2/jwuerzin/ALPaper/pmssm-expr-figures.tex
"""
from __future__ import annotations

import shutil
from pathlib import Path

import click

# Paper figure -> (equivalent basename or glob in the variant fig dir, reason
# when it cannot exist yet). An empty equivalent means "no counterpart".
PAPER_INVENTORY = [
    ("hit_rate_best_per_model.png",            "hit_rate_best_per_model.png", ""),
    ("hits_per_desired_best_per_model.png",    "hits_per_desired_best_per_model.png", ""),
    ("mse_best_per_model.png",                 "mse_best_per_model.png", ""),
    ("accuracy_best_per_model_static_random.png", "accuracy_best_per_model_static_random.png",
     "needs the accuracy pass (--compute-accuracy, GPU)"),
    ("mcmc_diagnostics_comparison.png",        "", "no emcee reference exists for this target"),
    ("input_overlay_inband.png",               "input_overlay_inband.png", ""),
    ("coverage_saturation.png",                "", "coverage support is built from the emcee reference"),
    ("coverage_saturation_pool.png",           "coverage_saturation_pool.png", ""),
    ("support_efficiency_40iter.png",          "support_efficiency_40iter.png", ""),
    ("support_efficiency_ext160.png",          "support_efficiency_ext160.png", ""),
    ("support_efficiency_probe20k.png",        "", "20k-batch probe sweep not submitted"),
    ("hit_rate_model_deep_gp.png",             "hit_rate_model_deep_gp_expr.png", ""),
    ("hit_rate_model_transformer.png",         "hit_rate_model_transformer_expr.png", ""),
    ("hit_rate_model_exact_gp.png",            "hit_rate_model_exact_gp_expr.png",
     "exact GP cells still running"),
    ("omega_overlay.png",                      "expr_overlay.png", ""),
    ("compute_vs_dataset.png",                 "compute_vs_dataset.png", ""),
    ("compute_vs_dataset_warm.png",            "compute_vs_dataset_warm.png",
     "per-warm-mode split needs both a warm- and a cold-pick model complete"),
    ("compute_vs_dataset_cold.png",            "compute_vs_dataset_cold.png",
     "per-warm-mode split needs both a warm- and a cold-pick model complete"),
    ("compute_per_iteration.png",              "compute_per_iteration.png", ""),
    ("hit_rate_oracle_comparison.png",         "", "oracle mode draws candidates from the emcee set"),
    ("accuracy_oracle_comparison_mcmc.png",    "", "oracle mode draws candidates from the emcee set"),
    ("pairwise_density_deep_gp_entropy_batch_warm.png",
     "pairwise_density_deep_gp_expr_entropy_batch_warm.png", ""),
    ("omega_vs_M_2_lsp_type.png",              "expr_vs_M_2_lsp_type.png", ""),
    ("prior_predictive_marginals.png",         "", "importance-sampling check of the emcee reference"),
    ("corner_random.png",                      "corner_random.png", ""),
    ("corner_mcmc.png",                        "", "no emcee reference exists for this target"),
    ("corner_deep_gp.png",                     "corner_deep_gp_expr.png", ""),
    ("corner_transformer.png",                 "corner_transformer_expr.png", ""),
    # The extension resumed the seed-1 runs in place, so these are single-seed
    # and were produced against the probe manifest built by
    # build_probe_manifest.py --from-manifest. They are placed by
    # GROUPED_SECTIONS, not here, so the two budgets stay side by side.
    ("probe_extended_yield.png",               "probe_extended_yield.png", ""),
    ("probe_extended_hitrate.png",             "probe_extended_hitrate.png", ""),
    ("mse_extended.png",                       "mse_extended.png", ""),
    ("probe_extended_inputs.png",              "probe_extended_inputs.png", ""),
    ("probe_extended_compute.png",             "probe_extended_compute.png", ""),
    ("probe_20k_yield.png",                    "", "20k-batch probe sweep not submitted"),
    ("probe_20k_hitrate.png",                  "", "20k-batch probe sweep not submitted"),
    ("probe_20k_inputs.png",                   "", "20k-batch probe sweep not submitted"),
    ("probe_20k_compute.png",                  "", "20k-batch probe sweep not submitted"),
    ("scaling_hitrate_vs_labelled.png",        "", "needs the 20k and extended probe budgets"),
    ("laplace_yield.png",                      "", "Laplace-acquisition variant sweep not submitted"),
    ("laplace_hitrate.png",                    "", "Laplace-acquisition variant sweep not submitted"),
    ("laplace_compute.png",                    "", "Laplace-acquisition variant sweep not submitted"),
    ("laplace_accuracy.png",                   "", "Laplace-acquisition variant sweep not submitted"),
    ("laplace_inputs_all.png",                 "", "Laplace-acquisition variant sweep not submitted"),
    ("laplace_inputs_inband.png",              "", "Laplace-acquisition variant sweep not submitted"),
]

# Figures pulled out of the counterpart flow into their own subsection, in this
# order, because they only mean anything read against each other: the same
# quantity at the standard 40-iteration budget and at the 160-iteration
# extension. Each group is fenced with \clearpage and \FloatBarrier so LaTeX
# cannot float a member of one group into the middle of the other. \clearpage is
# the load-bearing one: placeins' \FloatBarrier does not hold back full-width
# figure* floats, which most of these are.
GROUPED_SECTIONS = [
    ("Budget comparison: 40 iterations against the 160-iteration extension", [
        ("support_efficiency_40iter.png",
         "In-band support covered against budget, 40-iteration runs, five "
         "seeds. Budget is divided by the random scan's own size, so the "
         "horizontal gap to its curve is the cost ratio for equal coverage; "
         "the lower panel is that ratio read vertically at equal budget."),
        ("support_efficiency_ext160.png",
         "As above for the 160-iteration extension (single seed, resumed in "
         "place from the seed-1 runs), i.e. the same axes at roughly three "
         "times the labelled-set size."),
        ("probe_extended_hitrate.png",
         "Hit rate over the extension. Hits are acquired points with "
         "$|r-1| < \\mathrm{tol}$; the reference line is the random-scan "
         "prevalence."),
        ("probe_extended_yield.png",
         "Hits per requested point over the extension, so invalid simulator "
         "returns are charged to the method."),
        ("mse_extended.png",
         "Transformed-space MSE over the extension. Physical-space $R^2$ is "
         "not shown: $r$ reaches $\\sim 10^3$, so a cold-start prediction can "
         "overflow float32 on inversion."),
        ("probe_extended_compute.png",
         "Compute against labelled-set size over the extension, with the "
         "support-coverage panel built from the random-scan pool."),
        ("probe_extended_inputs.png",
         "Input-space marginals of the extended runs' in-band points against "
         "the random pool, $|r-1| < 0.1$ applied per arm."),
    ]),
]

# Extra families the paper shows only for selected cells but that are worth
# having in full here:
#   (glob, section title, caption stem, one-per-row?, repeat-if-already-placed?)
# The last field is True only for the per-model and per-strategy acquisition
# families, where the whole point is to see every setting side by side, so a
# member must not go missing just because the paper singles it out elsewhere.
# Everywhere else it is False, which stops the same corner or compute figure
# appearing twice in one document.
EXTRA_SECTIONS = [
    ("hit_rate_model_*.png", "Hit rate per model, all acquisition settings",
     "Hit rate, all strategy/warm settings of", False, True),
    ("hits_per_desired_model_*.png", "Hits per desired per model, all acquisition settings",
     "Hits per requested point, all strategy/warm settings of", False, True),
    ("hit_rate_strategy_*.png", "Hit rate per acquisition strategy, all models",
     "Hit rate, all models under", False, True),
    ("hits_per_desired_strategy_*.png", "Hits per desired per acquisition strategy, all models",
     "Hits per requested point, all models under", False, True),
    ("hit_rate_seeds_*.png", "Hit rate, individual seed replicas",
     "Unaggregated per-seed hit rate,", False, False),
    ("accuracy_best_per_model_*.png", "Classification accuracy at the boundary",
     "Binary accuracy either side of the boundary, evaluated on", False, False),
    ("val_r2_*.png", "Validation R-squared", "Validation R-squared,", False, False),
    ("static_r2_*.png", "Static random-set R-squared", "Static-set R-squared,", False, False),
    ("n_train_*.png", "Labelled-set growth", "Labelled-set size,", False, False),
    ("mse_*.png", "Mean squared error", "MSE,", False, False),
    ("compute_*.png", "Compute", "Compute,", False, False),
    ("corner_*.png", "Corner plots", "Input-space corner plot,", True, False),
    ("pairwise_density_*.png", "Pairwise input densities", "Pairwise input density,", True, False),
    ("*_overlay*.png", "Target-marginal overlays", "Target marginal,", False, False),
    ("*_vs_*_lsp_type.png", "Target vs mass parameter by LSP type",
     "Target against a mass parameter, coloured by LSP type,", False, False),
    ("*_vs_*_*.png", "Input-vs-target scatters", "Input against target,", False, False),
]

TEX_HEAD = r"""\RequirePackage{fix-cm}
\documentclass[pdftex,twocolumn,epjc3]{svjour3}
\RequirePackage[T1]{fontenc}
\smartqed
\RequirePackage{graphicx}
\RequirePackage{mathptmx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{longtable}
\RequirePackage{placeins}
\RequirePackage[colorlinks,citecolor=blue,urlcolor=blue,linkcolor=blue]{hyperref}
\journalname{Eur. Phys. J. C}

\graphicspath{{figures_expr/}}

\begin{document}

\title{Figure companion: active learning against the SModelS
       exclusion boundary in the electroweak pMSSM}
\titlerunning{Figure companion: exclusion-boundary target}
\author{Jonas W\"urzinger \and Dominik Vo\ss \and Lukas Heinrich}
\authorrunning{W\"urzinger, Vo\ss, Heinrich}
\institute{Physik-Department, Technische Universit\"at M\"unchen,
           James-Franck-Str.~1, 85748 Garching, Germany}
\date{Generated: GENDATE}
\maketitle
"""


# Figures whose content depends on a selection the reader cannot see. The note
# is appended to the caption; keep it factual.
CAPTION_NOTES = {
    "input_overlay_inband.png":
        "Restricted to in-band points, $|r-1| < 0.1$ (the 10\\% band around the "
        "exclusion boundary $r=1$), applied separately to each arm; densities "
        "are normalised so the arms' differing sizes are comparable. Model "
        "colours follow the hit-rate figures.",
    "input_overlay.png":
        "All acquired points, with no band cut: this shows where each loop "
        "looked, not where the in-band points are. Compare with the in-band "
        "variant. Densities are normalised per arm; model colours follow the "
        "hit-rate figures.",
    "expr_overlay.png":
        "Target marginals of each arm on shared log-log axes, with the "
        "$|r-1| < 0.1$ band shaded and the boundary $r=1$ marked. No band cut "
        "is applied to the curves themselves. Model colours follow the "
        "hit-rate figures.",
    "hit_rate_best_per_model.png":
        "Hits are points with $|r-1|/1 < \\mathrm{tol}$, one panel per "
        "tolerance; the reference line is the random-scan prevalence.",
    "hits_per_desired_best_per_model.png":
        "Same numerator as the hit rate, divided by the surrogate's REQUESTED "
        "point count, so invalid simulator returns are charged to the method. "
        "The random-scan reference is deflated by $p_{\\mathrm{valid}}=0.584$ "
        "to put both sides on a per-attempt footing.",
    "support_efficiency_40iter.png":
        "Budget is divided by the size of the dataset defining the support, so "
        "the horizontal gap between a curve and the random-scan curve is the "
        "factor by which generating more random points would cost more for the "
        "same coverage; the lower panel is the same comparison read vertically, "
        "as the ratio of each curve to the random scan's at equal budget, so "
        "above one means more support covered per simulator call than random "
        "generation buys. The AL budget counts training AND validation points, "
        "both of which were simulated; both axes count valid models, not "
        "attempts. The support is the 639 cells of the 12-bin quantile grid in "
        "$(M_1, M_2, \\mu)$ holding at least 20 in-band points of the pool half "
        "not used for any curve, in-band meaning $|r-1| < 0.1$. The random-scan "
        "curve is that held-out half, so no arm is scored against cells it "
        "defined. Model colours follow the hit-rate figures.",
    "compute_vs_dataset.png":
        "Coverage is the fraction of in-band support cells reached, on the "
        "12-bin quantile grid in $(M_1, M_2, \\mu)$ built from the random-scan "
        "pool's in-band half with at least 20 points per cell.",
}


def esc(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def fig_block(rel: str, caption: str, full_width: bool) -> str:
    note = CAPTION_NOTES.get(rel)
    if note:
        caption = f"{caption} {note}"
    env = "figure*" if full_width else "figure"
    width = "0.92\\textwidth" if full_width else "\\columnwidth"
    return (f"\\begin{{{env}}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width={width}]{{{rel}}}\n"
            f"  \\caption{{{caption}}}\n"
            f"\\end{{{env}}}\n\n")


@click.command()
@click.option("--fig-dir", default="/ptmp/jwuerzin/analysis/expr_runs", show_default=True,
              help="Directory holding the variant sweep's PNGs.")
@click.option("--out", default="/viper/u2/jwuerzin/ALPaper/pmssm-expr-figures.tex",
              show_default=True, help="LaTeX file to write.")
@click.option("--copy-figs/--no-copy-figs", default=True, show_default=True,
              help="Copy the PNGs next to the .tex (into figures_expr/), so the "
                   "document builds without reaching into /ptmp.")
@click.option("--gendate", default="", help="Timestamp string for the title page.")
def main(fig_dir, out, copy_figs, gendate):
    src = Path(fig_dir)
    out_p = Path(out)
    figdir = out_p.parent / "figures_expr"
    pngs = sorted(p.name for p in src.glob("*.png"))
    if not pngs:
        raise click.ClickException(f"no PNGs in {src}")

    if copy_figs:
        figdir.mkdir(parents=True, exist_ok=True)
        for n in pngs:
            shutil.copy2(src / n, figdir / n)

    used: set[str] = set()          # placed anywhere (drives the leftover section)
    section_used: set[str] = set()  # placed in a family section (avoids double-listing)
    body = []

    # ---- inventory table: every paper figure, available or not --------------
    avail = [(p, e) for p, e, _ in PAPER_INVENTORY if e and e in pngs]
    missing = [(p, e, r) for p, e, r in PAPER_INVENTORY if not (e and e in pngs)]
    body.append("\\onecolumn\n\\section{Inventory against the paper draft}\n")
    body.append("\\begin{longtable}{p{0.34\\textwidth}p{0.30\\textwidth}p{0.30\\textwidth}}\n"
                "\\hline\n paper figure & this document & status \\\\\n\\hline\n\\endhead\n")
    for p, e, r in PAPER_INVENTORY:
        if e and e in pngs:
            body.append(f"{esc(p)} & {esc(e)} & available \\\\\n")
        else:
            why = r or "not produced"
            body.append(f"{esc(p)} & --- & {esc(why)} \\\\\n")
    body.append("\\hline\n\\end{longtable}\n\n")
    body.append(f"\\noindent {len(avail)} of {len(PAPER_INVENTORY)} paper figures have an "
                f"equivalent here; {len(missing)} do not, for the reasons tabulated above.\n\n")
    body.append("\\twocolumn\n")

    # ---- the counterpart figures, in the paper's order ----------------------
    body.append("\\clearpage\n\\section{Counterparts of the paper's figures}\n")
    _grouped = {n for _t, ms in GROUPED_SECTIONS for n, _c in ms}
    for p, e, _ in PAPER_INVENTORY:
        if e and e in pngs and e not in used and e not in _grouped:
            used.add(e)
            body.append(fig_block(e, f"Counterpart of {esc(p)}.", False))

    # ---- grouped subsections ------------------------------------------------
    # Placed before the family sections so their members are already marked used
    # and cannot reappear; the barriers keep the two budgets visually separate.
    for title, members in GROUPED_SECTIONS:
        have = [(n, cap) for n, cap in members if n in pngs]
        if not have:
            continue
        body.append("\\clearpage\n\\FloatBarrier\n"
                    f"\\section{{{title}}}\n")
        for n, cap in have:
            used.add(n)
            section_used.add(n)
            body.append(fig_block(n, cap, True))
        body.append("\\FloatBarrier\n\\clearpage\n")
        click.echo(f"[report] grouped section {title!r}: {len(have)} of "
                   f"{len(members)} members present")

    # ---- the full families --------------------------------------------------
    # These families are shown COMPLETE, even when a member already appeared in
    # the counterpart section: the point of the per-model and per-strategy
    # sections is to hold every setting side by side, so dropping the ones the
    # paper happens to single out would defeat them. Only the leftover section
    # at the end excludes what has been placed.
    for pattern, title, stem, wide, repeat in EXTRA_SECTIONS:
        hits = [n.name for n in sorted(src.glob(pattern))]
        skip = section_used if repeat else used
        hits = [n for n in hits if n not in skip]
        if not hits:
            continue
        body.append(f"\\clearpage\n\\section{{{title}}}\n")
        for n in hits:
            used.add(n)
            section_used.add(n)
            tag = esc(n[:-4])
            body.append(fig_block(n, f"{stem} \\texttt{{{tag}}}.", wide))

    # ---- anything left over -------------------------------------------------
    leftover = [n for n in pngs if n not in used]
    if leftover:
        body.append("\\clearpage\n\\section{Further figures}\n")
        for n in leftover:
            body.append(fig_block(n, f"\\texttt{{{esc(n[:-4])}}}.", False))

    tex = TEX_HEAD.replace("GENDATE", gendate or "see file mtime") + "".join(body) + "\n\\end{document}\n"
    out_p.write_text(tex)
    click.echo(f"[report] {len(pngs)} figures, {len(used)} placed, "
               f"{len(leftover)} in the leftover section")
    click.echo(f"[report] wrote {out_p}")
    if copy_figs:
        click.echo(f"[report] copied PNGs to {figdir}")


if __name__ == "__main__":
    main()
