#!/bin/bash
# =============================================================================
# Quality metrics for the extended-budget and large-batch probes:
# neutralino composition, plus rank-normalised R-hat / ESS, so both appendices
# can be compared against the main body's tables.
#
# Runs on a LOGIN NODE, detached, because the SLURM job limit is saturated.
# Two consequences shape the design:
#
#   * Strictly sequential and single-threaded. Seven parallel readers once
#     exhausted the login node's process limit and OpenBLAS started failing in
#     pthread_create, so nothing here runs concurrently.
#   * Every stage is independent and writes its own output, with failures
#     tolerated, so a stage that dies still leaves the others' results behind.
#
# Launch detached so it survives logout (setsid puts it in a new session, so
# the SIGHUP that follows the shell's exit is never delivered):
#
#   setsid nohup bash scripts/run_probe_quality.sh > <log> 2>&1 < /dev/null &
#
# The UQ metrics (Var(z), rho, MCA, NLPD, CRPS, AUSE) are NOT here: they need a
# GPU to re-evaluate each checkpoint, so they have to wait for a SLURM slot.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EXT_DIR=/ptmp/jwuerzin/analysis/probe_extended
K20_DIR=/ptmp/jwuerzin/analysis/probe_20k

stage () {  # stage <label> <cmd...>
    local label="$1"; shift
    echo "=================================================================="
    echo "[stage] ${label}  started $(date '+%F %T')"
    echo "=================================================================="
    if "$@"; then
        echo "[stage] ${label}  OK $(date '+%F %T')"
    else
        echo "[stage] ${label}  FAILED rc=$? $(date '+%F %T')" >&2
    fi
}

# ---- refresh the manifests: runs are still in flight, so pick up new dirs ----
stage "manifest:extended" "${PY}" scripts/build_probe_manifest.py \
    --pattern '*ext160*' --out "${EXT_DIR}/manifest.csv" --sweep-id ext160
stage "manifest:20k" "${PY}" scripts/build_probe_manifest.py \
    --pattern '*_20k_*' --out "${K20_DIR}/manifest.csv" --sweep-id probe20k

# ---- composition: the slow stage (per-iteration worker ntuples) -------------
# Pool and emcee reference rows are skipped here; the main body already has
# them and re-reading 500k reference rows twice buys nothing.
stage "composition:extended" "${PY}" scripts/composition_fractions.py \
    --manifest "${EXT_DIR}/manifest.csv" --output-dir "${EXT_DIR}" \
    --baseline-data-dir "" --mcmc-data-dir "" --require-neutralino-lsp

stage "composition:20k" "${PY}" scripts/composition_fractions.py \
    --manifest "${K20_DIR}/manifest.csv" --output-dir "${K20_DIR}" \
    --baseline-data-dir "" --mcmc-data-dir "" --require-neutralino-lsp \
    --picks 'transformer:top_k:cold,dnn:top_k:cold,dnn_match_trafo:top_k:cold,deep_gp:top_k:warm'

# ---- R-hat / ESS across seed replicas --------------------------------------
# --data-dir is required even though the reference row is not what we are
# after here; keeping it identical to the main body's invocation means the
# MCMC row in each probe table is the same reference, so the AL rows are being
# compared against the same yardstick.
REF=/ptmp/jwuerzin/data/neutralino_v4

stage "diagnostics:extended" "${PY}" scripts/mcmc_diagnostics.py \
    --data-dir "${REF}" --mcmc-nwalkers 256 \
    --al-manifest "${EXT_DIR}/manifest.csv" --output-dir "${EXT_DIR}" \
    --require-neutralino-lsp

stage "diagnostics:20k" "${PY}" scripts/mcmc_diagnostics.py \
    --data-dir "${REF}" --mcmc-nwalkers 256 \
    --al-manifest "${K20_DIR}/manifest.csv" --output-dir "${K20_DIR}" \
    --require-neutralino-lsp \
    --al-picks 'transformer:top_k:cold,dnn:top_k:cold,dnn_match_trafo:top_k:cold,deep_gp:top_k:warm'

echo "=================================================================="
echo "[done] $(date '+%F %T')"
echo "=================================================================="
