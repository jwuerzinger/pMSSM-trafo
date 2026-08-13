#!/bin/bash
# ==============================================================================
# Generic AL smoke test for a given target and driver, WITH data generation.
#
# Exists because the second target axis (ExpR, the SModelS exclusion boundary)
# adds a SModelS step to the online generation, and that is the one part of the
# loop that cannot be verified offline: it needs a real Run3ModelGen subprocess,
# the SModelS results database resolved from the local cache, and a compute node
# with no internet. It also prints the per-iteration generation cost, which is
# the gate for whether a 40-iteration production run fits the 24 h limit.
#
# Submit from repo root:
#   source slurm/cluster.conf
#   AL_TARGET=ExpR AL_DRIVER=transformer \
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" \
#          --export=ALL slurm/test_al_target.sh
#
# Env:
#   AL_TARGET      DMRD | ExpR                        (default: ExpR)
#   AL_DRIVER      transformer | dnn | deep_gp | exact_gp | tabpfn
#                                                     (default: transformer)
#   AL_N_ITERATIONS / AL_N_SELECT / AL_GEN_WORKERS / AL_N_SAMPLES
#   AL_DATA_DIR    pool to ingest (defaults per target)
#   AL_TAG         suffix for the output dir           (default: smoke)
# ==============================================================================
#SBATCH --job-name=test_al_target
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
# See slurm/submit_al_*.sh: Run3ModelGen's setup.sh would otherwise point this
# at the repo checkout, which holds no database, and compute nodes cannot fetch.
export SMODELS_CACHEDIR="${SMODELS_CACHEDIR:-/ptmp/jwuerzin/cache/smodels}"

AL_TARGET="${AL_TARGET:-ExpR}"
AL_DRIVER="${AL_DRIVER:-transformer}"
AL_TAG="${AL_TAG:-smoke}"

case "${AL_TARGET}" in
    ExpR) DEFAULT_DATA_DIR="${CLUSTER_DATA_DIR}/260804"; MCMC_ARGS="--no-mcmc-eval";;
    DMRD) DEFAULT_DATA_DIR="${CLUSTER_DATA_DIR}/18387358"
          MCMC_ARGS="--mcmc-data-dir ${CLUSTER_DATA_DIR}/neutralino_v4 --mcmc-max-samples 50000";;
    *) echo "[error] unsupported AL_TARGET: ${AL_TARGET}" >&2; exit 1;;
esac
AL_DATA_DIR="${AL_DATA_DIR:-${DEFAULT_DATA_DIR}}"

case "${AL_DRIVER}" in
    transformer) SCRIPT="active_learning.py";        MODEL_ARGS="--y-transform log";;
    dnn)         SCRIPT="active_learning_dnn.py";    MODEL_ARGS="--y-transform log";;
    deep_gp)     SCRIPT="active_learning_gp.py";     MODEL_ARGS="--model-type deep_gp --num-inducing-max 256 --kernel RBF --use-ard --learning-rate 1e-3 --jitter 1e-3";;
    exact_gp)    SCRIPT="active_learning_gp.py";     MODEL_ARGS="--model-type exact_gp --kernel RBF --use-ard --learning-rate 1e-3 --jitter 1e-3";;
    tabpfn)      SCRIPT="active_learning_tabpfn.py"; MODEL_ARGS="--n-ensemble-samples 8";;
    *) echo "[error] unsupported AL_DRIVER: ${AL_DRIVER}" >&2; exit 1;;
esac

PIXI_ENV="${PIXI_ENV:-rocm}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"

OUT="/ptmp/jwuerzin/output/smoke_${AL_TARGET}_${AL_DRIVER}_${AL_TAG}_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo " target=${AL_TARGET} driver=${AL_DRIVER} (${SCRIPT})"
echo " data_dir=${AL_DATA_DIR}"
echo " SMODELS_CACHEDIR=${SMODELS_CACHEDIR}"
echo " out=${OUT}"
echo " node=$(hostname)  started=$(date)"
echo "=========================================="

"${PYTHON}" "${SCRIPT}" \
    --target "${AL_TARGET}" \
    ${MCMC_ARGS} \
    --data-dir "${AL_DATA_DIR}" \
    --n-samples "${AL_N_SAMPLES:-2000}" \
    --n-iterations "${AL_N_ITERATIONS:-2}" \
    --n-select "${AL_N_SELECT:-100}" \
    --n-candidates "${AL_N_CANDIDATES:-50000}" \
    --entropy-pool-size 2000 \
    --tolerance-sampling 1.0 \
    --static-eval-size "${AL_STATIC_EVAL_SIZE:-20000}" \
    --gen-workers "${AL_GEN_WORKERS:-20}" \
    --selection-strategy "${AL_SELECTION_STRATEGY:-entropy_batch}" \
    --output-dir "${OUT}" \
    --gpu-ids 0 \
    --generate-data \
    ${MODEL_ARGS} \
    ${AL_EXTRA_ARGS:-}

echo "=========================================="
echo " Done: $(date)"
echo "----- generation-phase timings -----------"
# Each iteration logs "Generation target: N valid models" then, once the workers
# return, "Generation target reached" or "After attempt". The wall gap between
# them is the cost this smoke test exists to measure.
grep -E "Generation target|After attempt|valid models from ntuple|Starting model generation|absent from ntuple" \
    "${OUT}/active_learning.log" | tail -40 || true
echo "----- ingest line ------------------------"
grep -m1 "Filter (" "${OUT}/active_learning.log" || true
grep -m1 "  target:" "${OUT}/active_learning.log" || true
echo "=========================================="
