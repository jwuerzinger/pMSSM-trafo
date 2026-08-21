"""End-of-iteration housekeeping: summarise, then pack the simulator workspaces.

Why this runs inside the loop
-----------------------------
Each AL iteration leaves its Run3ModelGen workspaces under
``iteration_NNN/worker_*`` and ``retry_*``: measured on a benchmark iteration,
**6,213 files and 94 MB**, of which 5,550 files are read by nothing. Nothing in
the loop removed them, so a 40-iteration run reached 5.8 GB and 227,000 files.

That is an inode problem rather than a space problem. ``/ptmp`` (viper_ptmp2)
has no per-user file quota but a hard filesystem ceiling that does not grow
(``mmlsfs``: ``--inode-limit 629145600``, ``--auto-inode-limit no``), of which
about 268 million were free on 2026-08-21. A 200-iteration campaign left
untouched would have claimed most of that headroom.

Doing it here rather than from a janitor means the process that owns the
directory is the one that packs it, so no external sweep can ever race a live
iteration. It runs after ``save_state``, so the iteration's training data is
already durable in ``state.pt`` before anything is touched.

What survives, and why
----------------------
``ntuple.*.root`` stays on disk, moved into a flat ``iteration_NNN/ntuples/``.
``scripts/composition_fractions.py`` reads these with
``iteration_dir.rglob("ntuple.*.root")``, and the rename keeps that glob
matching (all 40 workers write the identical name ``ntuple.0.0.root``, so the
worker index has to go into the filename for a flat directory to be possible at
all). Flattening matters: keeping them in place would mean keeping 841
directories per iteration alive to hold 40 files.

Everything else is packed into ``iteration_NNN/debris.tar`` and removed. The
``*.slha.py`` SModelS outputs go into the tar too, which would otherwise cost
``best_analysis_arms.py``, ``best_analysis_from_smodels.py`` and
``mode_switch_diagnostic.py`` their input, so the information they extract is
computed here first and written to ``iteration_NNN/smodels_best_analysis.json``
plus a summary line in the run log. The offline study then reads a summary
instead of re-``exec``-ing 623 files per iteration.

Nothing here may break a run. Every step is wrapped: a failure is logged and
the iteration keeps its workspaces, which is the harmless direction.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Neutralino composition branches, as the ntupler writes them.
_FRAC_BRANCHES = ("SP_LSP_Bino_frac", "SP_LSP_Wino_frac", "SP_LSP_Higgsino_frac")

# Per-point results kept in the summary, ranked by r_expected. Enough for the
# winner and for the gap to the next distinct AnalysisID; not enough to
# reproduce best_analysis_from_smodels.ntuple_style_winner, which walks every
# result and which its own docstring says is never the measurement.
RANK_KEEP = 8


def _debris_dirs(iter_dir: Path):
    return sorted([p for p in iter_dir.glob("worker_*") if p.is_dir()] +
                  [p for p in iter_dir.glob("retry_*") if p.is_dir()])


# ── the SModelS record that replaces 623 executable files ────────────────────
def _read_winner(path):
    """The winning expected-r result of one SModelS output file.

    SModelS writes these as executable Python (``smodelsOutput = {...}``), which
    is how Run3ModelGen's ntupler reads them, so exec is the format's intended
    reader rather than a shortcut. Mirrors ``read_winner`` in
    scripts/best_analysis_from_smodels.py, minus its band cut: the cut belongs to
    the offline study, so the record keeps every point and lets the study choose.
    """
    g: dict = {}
    try:
        with open(path) as fh:
            exec(fh.read(), g)                                  # noqa: S102
    except Exception:
        return None
    er = (g.get("smodelsOutput") or {}).get("ExptRes")
    if not er:
        return None
    cand = [r for r in er if r.get("r_expected") is not None]
    if not cand:
        return None
    best = max(cand, key=lambda r: r["r_expected"])
    # The ranking, not only the winner: mode_switch_diagnostic.py measures the
    # gap from the top result to the next one and to the next DISTINCT
    # AnalysisID, so a winner-only record would not serve it. Truncated to
    # RANK_KEEP because a point carries ~33 results and the full list would make
    # this file an order of magnitude larger for no measurement anyone takes.
    ranked = sorted(cand, key=lambda r: -r["r_expected"])[:RANK_KEEP]
    return {"analysis": best.get("AnalysisID"),
            "txnames": list(best.get("TxNames") or []),
            "r_expected": float(best["r_expected"]),
            "n_results": len(er),
            "ranked": [[float(r["r_expected"]), r.get("AnalysisID", "?"),
                        r.get("theory prediction (fb)"),
                        r.get("expected upper limit (fb)")] for r in ranked]}


def summarise_smodels(iter_dir: Path, logger, true_value=1.0, tolerance=0.10):
    """Write the per-point SModelS winners to JSON and log the shares.

    Returns the summary dict, or None when there is nothing to read.
    """
    files = []
    for d in _debris_dirs(iter_dir):
        files.extend(sorted(d.rglob("*.slha.py")))
    if not files:
        return None
    t0 = time.time()
    points, skipped = [], 0
    for p in files:
        rec = _read_winner(p)
        if rec is None:
            skipped += 1
            continue
        # Keep the source path: it is the only link back to the model once the
        # workspaces are inside the tar.
        rec["source"] = str(p.relative_to(iter_dir))
        points.append(rec)
    if not points:
        return None
    r = np.array([p["r_expected"] for p in points], dtype=float)
    in_band = np.abs(r / true_value - 1.0) <= tolerance
    shares = Counter(p["analysis"] for p, m in zip(points, in_band) if m)
    summary = {
        "n_points": len(points), "n_unreadable": skipped,
        "true_value": float(true_value), "tolerance": float(tolerance),
        "n_in_band": int(in_band.sum()),
        "r_expected_median": float(np.median(r)),
        "r_expected_max": float(r.max()),
        "in_band_analysis_counts": dict(shares.most_common()),
        "definition": "max r_expected over ExptRes per point, as in "
                      "scripts/best_analysis_from_smodels.py:read_winner; "
                      f"'ranked' holds the top {RANK_KEEP} results per point as "
                      "[r_expected, AnalysisID, theory_pred_fb, exp_ul_fb]",
        "rank_keep": RANK_KEEP,
    }
    out = iter_dir / "smodels_best_analysis.json"
    out.write_text(json.dumps({"summary": summary, "points": points}, indent=1))
    top = ", ".join(f"{a}={n}" for a, n in shares.most_common(4)) or "none in band"
    logger.info(f"[housekeeping] SModelS best analysis: {len(points)} points, "
                f"{int(in_band.sum())} within {tolerance:.0%} of "
                f"{true_value:g}; leading: {top} "
                f"({time.time() - t0:.1f}s) -> {out.name}")
    return summary


# ── the composition record, from the ntuples that stay on disk ───────────────
def summarise_composition(ntuples, logger, target_branch=None,
                          true_value=None, tolerance=0.10):
    """Log the LSP composition shares of this iteration's generated models.

    The ntuples themselves stay on disk, so this is a convenience for the
    offline study rather than the only record. Uses the same definition as the
    composition figure: the ntupler's -1 sentinel for a non-neutralino LSP is
    NaN'd out first, or a raw -1 survives the isfinite check in
    ``classify_lsp_type`` and is labelled "mixed".
    """
    if not ntuples:
        return None
    try:
        import uproot                                          # noqa: PLC0415
        from .visualization import LSP_TYPE_NAMES, classify_lsp_type
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[housekeeping] composition skipped: {exc}")
        return None
    fr, tv = [], []
    for p in ntuples:
        try:
            t = uproot.open(p)["susy"]
            cols = {b: t[b].array(library="np") for b in _FRAC_BRANCHES}
            fr.append(np.stack([cols[b] for b in _FRAC_BRANCHES], axis=1))
            if target_branch:
                tv.append(t[target_branch].array(library="np"))
        except Exception:                                      # noqa: BLE001
            continue
    if not fr:
        return None
    F = np.concatenate(fr).astype(np.float64)
    F[(F < 0).any(axis=1)] = np.nan          # the -1 sentinel
    labels = classify_lsp_type(F)
    ok = labels >= 0
    summary = {"n_all": int(len(labels)), "n_neutralino": int(ok.sum())}
    for k, name in LSP_TYPE_NAMES.items():
        summary[name] = (float((labels[ok] == k).mean()) if ok.any()
                         else float("nan"))
    if tv and true_value is not None:
        y = np.concatenate(tv).astype(np.float64)
        n = min(len(y), len(labels))
        band = np.abs(y[:n] / true_value - 1.0) <= tolerance
        okb = band & ok[:n]
        summary["n_in_band"] = int(band.sum())
        for k, name in LSP_TYPE_NAMES.items():
            summary[f"in_band_{name}"] = (float((labels[:n][okb] == k).mean())
                                          if okb.any() else float("nan"))
    logger.info("[housekeeping] N1 composition of {n_neutralino}/{n_all} "
                "neutralino-LSP models: bino {bino:.3f} wino {wino:.3f} "
                "higgsino {higgsino:.3f} mixed {mixed:.3f}".format(**summary))
    if "n_in_band" in summary:
        logger.info(f"[housekeeping] in-band ({summary['n_in_band']} models): "
                    f"bino {summary['in_band_bino']:.3f} "
                    f"wino {summary['in_band_wino']:.3f} "
                    f"higgsino {summary['in_band_higgsino']:.3f} "
                    f"mixed {summary['in_band_mixed']:.3f}")
    return summary


# ── keep the ntuples, pack the rest ──────────────────────────────────────────
def _rescue_ntuples(iter_dir: Path, dirs, logger):
    """Move every ntuple into a flat iteration_NNN/ntuples/, return new paths.

    All workers write the identical basename ``ntuple.0.0.root``, so the worker
    (and retry) index goes into the filename. The name still starts with
    ``ntuple.`` and ends ``.root``, which is what keeps
    ``composition_fractions.py``'s ``rglob("ntuple.*.root")`` matching.
    """
    dest = iter_dir / "ntuples"
    moved = []
    for d in dirs:
        for p in sorted(d.rglob("ntuple.*.root")):
            rel = p.relative_to(iter_dir)
            # worker_07/scan/ntuple.0.0.root      -> ntuple.worker_07.0.0.root
            # retry_001/worker_03/scan/ntuple.0.0.root
            #                        -> ntuple.retry_001-worker_03.0.0.root
            tag = "-".join(part for part in rel.parts[:-1]
                           if part.startswith(("worker_", "retry_")))
            new = dest / f"ntuple.{tag}.{p.name[len('ntuple.'):]}"
            dest.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(new))
                moved.append(new)
            except Exception as exc:                           # noqa: BLE001
                logger.warning(f"[housekeeping] could not move {rel}: {exc}")
    return moved


def pack_iteration_debris(iter_dir: Path, logger):
    """Tar iteration_NNN/{worker_*,retry_*} and remove them. Ntuples stay out.

    The removal is gated on reading the archive back and counting its regular
    members against the count measured on disk. "the tar is not empty" is not a
    verification: a tar truncated by a full filesystem satisfies it, and the
    source tree, the only copy, would be removed anyway.
    """
    dirs = _debris_dirs(iter_dir)
    if not dirs:
        return None
    tarball = iter_dir / "debris.tar"
    if tarball.exists():
        return None
    kept = _rescue_ntuples(iter_dir, dirs, logger)
    n_disk = sum(len(f) for d in dirs for _r, _sub, f in os.walk(d))
    t0 = time.time()
    try:
        with tarfile.open(tarball, "w") as tf:
            for d in dirs:
                tf.add(d, arcname=d.name)
        with tarfile.open(tarball, "r") as tf:
            packed = sum(1 for m in tf.getmembers() if m.isfile())
        if packed != n_disk:
            raise OSError(f"packed {packed} files, expected {n_disk}")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[housekeeping] workspaces KEPT, pack failed: "
                       f"{type(exc).__name__}: {exc}")
        try:
            tarball.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    removed = 0
    for d in dirs:
        errs: list = []
        shutil.rmtree(d, onerror=lambda *a: errs.append(a[1]))
        if errs or d.exists():
            logger.warning(f"[housekeeping] {d.name} not fully removed "
                           f"({len(errs)} error(s)); its data is in "
                           f"{tarball.name}")
        else:
            removed += 1
    size = tarball.stat().st_size / 2 ** 20
    logger.info(f"[housekeeping] packed {removed}/{len(dirs)} workspace dirs "
                f"({n_disk:,} files, {size:.0f} MiB) into {tarball.name}; "
                f"kept {len(kept)} ntuple(s) in ntuples/ "
                f"({time.time() - t0:.1f}s)")
    return {"dirs": len(dirs), "files": n_disk, "ntuples_kept": len(kept),
            "tar_mib": size}


def finalise_iteration(iter_dir, logger, target_branch=None, true_value=None,
                       tolerance=0.10, enabled=True):
    """Summarise this iteration's simulator output, then pack the workspaces.

    Call AFTER save_state: the training data is durable by then, so nothing here
    can cost an iteration. Order matters, the SModelS summary must be taken
    before the *.slha.py files move into the tar.

    Never raises. A failure leaves the workspaces in place, which is the
    harmless direction.
    """
    if not enabled:
        return
    iter_dir = Path(iter_dir)
    if not _debris_dirs(iter_dir):
        # Say so rather than returning in silence. A run with
        # --no-generate-data has no workspaces at all, and an operator reading
        # the log needs to be able to tell that apart from a call site that
        # never ran.
        logger.info("[housekeeping] no simulator workspaces in "
                    f"{iter_dir.name}; nothing to summarise or pack")
        return
    try:
        summarise_smodels(iter_dir, logger,
                          true_value=1.0 if true_value is None else true_value,
                          tolerance=tolerance)
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[housekeeping] SModelS summary failed: "
                       f"{type(exc).__name__}: {exc}")
    try:
        info = pack_iteration_debris(iter_dir, logger)
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[housekeeping] packing failed: "
                       f"{type(exc).__name__}: {exc}")
        return
    if info is None:
        return
    try:
        summarise_composition(sorted((iter_dir / "ntuples").glob("ntuple.*.root")),
                              logger, target_branch=target_branch,
                              true_value=true_value, tolerance=tolerance)
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(f"[housekeeping] composition summary failed: "
                       f"{type(exc).__name__}: {exc}")


# ── reading the record back ──────────────────────────────────────────────────
def load_iteration_smodels(iter_dir):
    """Per-point SModelS winners for one iteration, from whichever source exists.

    Returns a list of records shaped like ``_read_winner``'s output, or [].

    Completed runs are unaffected by any of this: their ``worker_*``/``retry_*``
    trees are intact, so ``scripts/best_analysis_arms.py``,
    ``scripts/best_analysis_from_smodels.py`` and
    ``scripts/mode_switch_diagnostic.py`` keep reading the loose ``*.slha.py``
    files exactly as before. This helper is the route for runs whose workspaces
    have been packed: prefer the loose files where they exist, fall back to the
    JSON where they do not, so one call serves both.

    Not yet wired into those three scripts. What it cannot serve is
    ``best_analysis_from_smodels.ntuple_style_winner``, which walks every result
    of every point rather than the top RANK_KEEP, and which its own docstring
    calls "never the measurement".
    """
    iter_dir = Path(iter_dir)
    loose = []
    for d in _debris_dirs(iter_dir):
        loose.extend(sorted(d.rglob("*.slha.py")))
    if loose:
        out = []
        for p in loose:
            rec = _read_winner(p)
            if rec is not None:
                rec["source"] = str(p.relative_to(iter_dir))
                out.append(rec)
        return out
    cached = iter_dir / "smodels_best_analysis.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text()).get("points", [])
        except Exception:                                      # noqa: BLE001
            return []
    return []
