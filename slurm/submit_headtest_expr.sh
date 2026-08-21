#!/bin/bash
# ==============================================================================
# Slurm job: single-seed, 40-iteration ExpR runs for the two new acquisition
# arms, one job per (model, arm).
#
#   arm "tolrand"  regression head, --selection-strategy tol_only_random:
#                  the tolerance cut then a uniform draw, i.e. mean-guided
#                  acquisition with the uncertainty never consulted. Offline
#                  (Stage A) this beat every sigma ranking on ExpR, so this run
#                  measures it in the real loop, with simulator dropout and the
#                  iterative feedback a static test cannot capture.
#   arm "bald"     classification head, --selection-strategy bald: train on the
#                  verdict 1[log r > 0], acquire by committee disagreement
#                  I(y;theta) = H[p_bar] - E H[p]. Offline this reached 3.3x
#                  (DNN) and 6.3x (Transformer) the random band rate, against
#                  0.4-0.5x for every regression ranking. On the GPs the same
#                  arm tests the opposite hypothesis: their regression posterior
#                  variance is target-independent pure geometry, which is why
#                  they survive the kinks in r_exp = max_a r^(a), and a
#                  boundary-anchored score deliberately discards that geometry.
#                  If classification COSTS the GPs, the head is a workaround for
#                  bad uncertainty rather than an improvement in itself.
#
# Each model runs in its published best warm/cold configuration: cold for the
# neural surrogates, warm for the GPs, and TabPFN has no such axis.
#
# Everything the run needs is set INSIDE this script on purpose. The submitting
# shell has SBATCH_EXPORT=NONE, so relying on exported AL_* variables risks a
# job silently falling back to the DMRD defaults, which would be a wrong 12-hour
# run that looks like a right one.
#
# Usage:
#   sbatch slurm/submit_headtest_expr.sh <model> <arm>
#     model : transformer | dnn | dnnmatch | exactgp | deepgp | tabpfn | lsqgp
#             exactgp + bald/clsent uses the Laplace GP classifier (R&W 3.4);
#             lsqgp is the least-squares classifier (R&W 6.5) on the exact GP.
#     arm   : tolrand | bald | clsent   (bald/clsent: neural surrogates only)
# ==============================================================================
#SBATCH --job-name=hx
#SBATCH --partition=apu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# ONE GPU on purpose. A 2-GPU request is routed to the EXCLUSIVE apu partition
# regardless of the --partition=apu1 above, where it queues behind every
# multi-node production job; a 1-GPU request stays on the shared apu1 and
# backfills within minutes. The cost is that the AL and baseline models train
# sequentially rather than on a GPU each, which roughly doubles the training
# time per iteration and is far cheaper than the queue wait it avoids.
#SBATCH --gres=gpu:1
# 100G, not 128G: a 1-GPU request on a 220G node may not ask for more than half
# the node's memory ("requested only 1 of two apus but more than 1/2 of memory
# of the node"), and 128G trips that. The Laplace classifier's peak is ~30G at
# the n this loop reaches, so 100G is not the binding constraint.
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

MODEL="${1:?model required: transformer|dnn|dnnmatch|exactgp|deepgp|tabpfn}"
ARM="${2:?arm required: tolrand|bald}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)
SEED="${HX_SEED:-1}"

# ---- ExpR essentials (see the ExpR project notes) ----------------------------
# The SModelS database must be found or every generation worker dies trying to
# download it; both the pickle and its descriptor live in this directory.
export SMODELS_CACHEDIR="${SMODELS_CACHEDIR:-/ptmp/jwuerzin/cache/smodels}"
POOL_DIR="/ptmp/jwuerzin/data/260804"

# Which verdict head a model can carry is a property of its inference scheme,
# not a preference. A Bernoulli likelihood is non-conjugate, so:
#   neural, deep GP  -> "classification": BCE logits / a true Bernoulli GP
#                       classifier. The deep GP is already variational, so the
#                       likelihood swap leaves its inference untouched.
#   exact GP         -> "lsq_classification": regress the +-1 verdict under the
#                       Gaussian likelihood and read it through a probit link.
#                       This keeps exact conjugate inference, so the arm differs
#                       from its regression counterpart in the training target
#                       ALONE. A Bernoulli exact GP would change the head and
#                       the inference scheme at once and confound the comparison.
case "${MODEL}" in
    # The canonical single-layer GP classifier, Rasmussen & Williams section
    # 3.4: probit likelihood, Laplace-approximated posterior (their Algorithms
    # 3.1 and 3.2). This is also the model Houlsby et al. (2011) derive BALD
    # for, so model and acquisition rest on the same source. It overrides the
    # submit script's own --model-type, which click resolves last-wins.
    exactgp) CLS_ARGS="--model-type laplace_gpc --head classification" ;;
    # R&W section 6.5, least-squares classification: regress the +-1 label under
    # the Gaussian likelihood. Kept as a separate arm rather than as the exact
    # GP's classifier because it answers a different question: it holds exact
    # conjugate inference fixed and changes only the training target, which
    # Laplace cannot do.
    lsqgp)   CLS_ARGS="--head lsq_classification" ;;
    *)       CLS_ARGS="--head classification" ;;
esac

case "${ARM}" in
    tolrand) ARM_ARGS="--selection-strategy tol_only_random" ;;
    bald)    ARM_ARGS="${CLS_ARGS} --selection-strategy bald" ;;
    # The other classification score: H[p_bar] of the mean probability, i.e.
    # "committee mean nearest 0.5", which is what the literature uses. Offline
    # it trailed BALD by 2-3x, so both are run rather than one being assumed.
    clsent)  ARM_ARGS="${CLS_ARGS} --selection-strategy cls_entropy" ;;
    *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

# Warm/cold: cold for the neural surrogates, warm for the GPs. The per-model
# submit scripts hardcode --warm-starting, so cold is expressed by appending the
# negated flag, which click resolves last-wins.
case "${MODEL}" in
    transformer) SCRIPT=submit_al_transformer.sh; MODEL_ARGS="--no-warm-starting" ;;
    dnn)         SCRIPT=submit_al_dnn.sh;         MODEL_ARGS="--no-warm-starting" ;;
    dnnmatch)    SCRIPT=submit_al_dnn.sh;         MODEL_ARGS="--no-warm-starting --num-layers 3 --dim-feedforward 400" ;;
    exactgp)     SCRIPT=submit_al_gp_exact.sh;    MODEL_ARGS="" ;;
    lsqgp)       SCRIPT=submit_al_gp_exact.sh;    MODEL_ARGS="" ;;
    deepgp)      SCRIPT=submit_al_gp_deep.sh;     MODEL_ARGS="" ;;
    tabpfn)      SCRIPT=submit_al_tabpfn.sh;      MODEL_ARGS="" ;;
    *) echo "unknown model ${MODEL}" >&2; exit 2 ;;
esac

# The GP drivers honour only AL_EXTRA_ARGS and AL_OUTPUT_DIR, while the neural
# ones honour fifteen AL_* variables including AL_DATA_DIR. Exporting AL_DATA_DIR
# therefore did NOTHING for a GP run, which read the hardcoded DMRD pool
# (18387358) and died on the missing ExpR branch after three minutes. The GP
# scripts now honour AL_DATA_DIR too, and --data-dir is ALSO passed explicitly
# below: this wrapper's contract is that the run's configuration lives here, not
# in whatever the per-model script happens to respect.
#
# The two drivers do not take the same flags, and AL_EXTRA_ARGS is passed to
# them verbatim: active_learning_gp.py has no --target-value and no
# --y-transform, because it reads both out of TARGET_CONFIG[target] (ExpR ->
# true value 1.0, log transform). Passing the neural spelling to a GP is not a
# silently wrong run but an instant "No such option" exit, which is why this is
# keyed off the driver rather than assumed common.
# The Laplace classifier pays O(n^3/6) per NEWTON step and takes several Newton
# steps per hyperparameter step, so the regression arm's 10000-epoch budget does
# not transfer. The first attempt used 400 steps at lr 5e-3, sized from an
# estimate that turned out to be 70x too pessimistic (185 s/step predicted by a
# cubic extrapolation from n=1600, against 2.755 s measured at n=14000). At that
# budget the fit did not move: the approximate log marginal likelihood improved
# 1.7% in 50 steps, validation NLPD was flat, and the model sat at the
# majority class (accuracy 0.7114 against a majority rate of 0.7113).
#
# 3000 steps at lr 1e-2 is affordable on the measured cost (~18 min per fit at
# the n this loop reaches by iteration 40, less earlier) and gives 19 ARD
# lengthscales a real chance from a cold start. This is still a different
# training protocol from the regression arm's 10000/1e-3/100, so a
# regression-vs-classification comparison on the exact GP is confounded by
# optimisation budget and has to say so.
if [[ "${MODEL}" == "exactgp" && "${ARM}" =~ ^(bald|clsent)$ ]]; then
    MODEL_ARGS="${MODEL_ARGS} --epochs 3000 --patience 200 --learning-rate 1e-2"
fi

case "${SCRIPT}" in
    submit_al_gp_*.sh) COMMON="--target ExpR --no-mcmc-eval --seed ${SEED} --data-dir ${POOL_DIR}" ;;
    *)                 COMMON="--target ExpR --target-value 1.0 --no-mcmc-eval --y-transform log --seed ${SEED}" ;;
esac

if [[ "${ARM}" =~ ^(bald|clsent)$ && "${MODEL}" == "tabpfn" ]]; then
    echo "[fatal] a classification TabPFN would need TabPFNClassifier, and with" >&2
    echo "        no committee it could only produce H[p_bar], never BALD." >&2
    exit 2
fi

export AL_DATA_DIR="${POOL_DIR}"
export AL_N_ITERATIONS=40
export AL_OUTPUT_DIR="/ptmp/jwuerzin/output/headtest_${MODEL}_${ARM}_seed${SEED}_${TS}"
export AL_EXTRA_ARGS="${COMMON} ${ARM_ARGS} ${MODEL_ARGS}"

echo "=========================================="
echo " Job:    ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:   $(hostname)"
echo " Model:  ${MODEL} (${SCRIPT})   Arm: ${ARM}   Seed: ${SEED}"
echo " Extra:  ${AL_EXTRA_ARGS}"
echo " Pool:   ${AL_DATA_DIR}"
echo " Out:    ${AL_OUTPUT_DIR}"
echo " Cache:  ${SMODELS_CACHEDIR}"
echo "=========================================="

# The per-model script carries the environment setup, GPU discovery and the
# python invocation. Run it as a child rather than sourcing it: our exports
# reach it through the environment, while its own `set -euo pipefail` and any
# positional-argument use stay isolated from this wrapper's $1/$2.
exec bash "slurm/${SCRIPT}"
