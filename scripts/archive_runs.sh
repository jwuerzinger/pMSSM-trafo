#!/bin/bash
# ==============================================================================
# archive_runs.sh — back up the irreplaceable part of every AL run directory
# into per-run tarballs on non-purged project storage.
#
# Why: /ptmp is auto-purged (files unaccessed for ~12 weeks are deleted); it
# has already eaten most exact_gp and deep_gp per-iteration checkpoints. The
# home filesystem is not purged but has a tight INODE quota, so raw rsync of
# the run trees (worker_*/retry_* SPheno workspaces = many thousands of small
# files per run) is not an option. One tar per run costs one inode.
#
# Archived per run (the curated, unregenerable subset):
#   state.pt                                cumulative training data (X, Y, F)
#   active_learning.log                     provenance incl. p_valid line
#   accuracy_trajectory.json                accuracy cache (if present)
#   iteration_*/{al,baseline}_model_checkpoint.pt   per-iteration weights
# Deliberately excluded: worker_*/retry_* simulation debris (regenerable),
# gifs/ and plots/ (re-renderable from state.pt).
#
# Idempotent: an existing tarball is only rewritten when the run's state.pt
# is newer than the tarball, so re-running after a sweep only archives new or
# updated runs. Run after every sweep completes:
#
#   bash scripts/archive_runs.sh
#
# Optional args: SRC dir, DEST dir, and a glob PATTERN over run-dir names
# (default "*"). The bundle jobs pass their own name prefix as PATTERN so
# concurrently finishing bundles each archive only their own runs:
#
#   bash scripts/archive_runs.sh /ptmp/jwuerzin/output \
#        /viper/u2/jwuerzin/pmssm-archive/runs "active_learning_dnn_top_k_cold_seed*"
# ==============================================================================
set -uo pipefail

SRC="${1:-/ptmp/jwuerzin/output}"
DEST="${2:-/viper/u2/jwuerzin/pmssm-archive/runs}"
PATTERN="${3:-*}"

mkdir -p "$DEST"
cd "$SRC" || exit 1

n_new=0 n_skip=0 n_empty=0
for d in ${PATTERN%/}/; do
    name=${d%/}
    [[ -d "$name" ]] || continue
    out="$DEST/$name.tar"
    ref="$name/state.pt"
    [[ -f "$ref" ]] || ref="$name"           # fall back to dir mtime
    if [[ -f "$out" && "$out" -nt "$ref" ]]; then
        n_skip=$((n_skip + 1))
        continue
    fi
    # Collect members that actually exist (tar errors on missing operands).
    members=()
    for f in "$name/state.pt" "$name/active_learning.log" \
             "$name/accuracy_trajectory.json"; do
        [[ -f "$f" ]] && members+=("$f")
    done
    while IFS= read -r f; do members+=("$f"); done \
        < <(find "$name" -maxdepth 2 -path "*/iteration_*/*_model_checkpoint.pt" 2>/dev/null | sort)
    if [[ ${#members[@]} -eq 0 ]]; then
        n_empty=$((n_empty + 1))
        continue
    fi
    if tar -cf "$out.tmp.$$" "${members[@]}" 2>/dev/null; then
        mv "$out.tmp.$$" "$out"
        n_new=$((n_new + 1))
    else
        rm -f "$out.tmp.$$"
        echo "[archive] WARN: tar failed for $name" >&2
    fi
done

echo "[archive] done: $n_new archived, $n_skip up-to-date, $n_empty empty/skipped"
echo "[archive] dest: $DEST ($(ls "$DEST" | wc -l) tarballs, $(du -sh "$DEST" 2>/dev/null | cut -f1))"
