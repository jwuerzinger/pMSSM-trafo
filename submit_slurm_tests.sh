#!/bin/bash
# Submit all architecture test jobs to Slurm.
# Run from repo root: bash submit_slurm_tests.sh

set -euo pipefail

mkdir -p /raven/u/jwuerzin/pMSSM-trafo/logs

# Transformer
sbatch slurm/test_al_transformer.sh

# ExactGP
sbatch slurm/test_al_gp_exact.sh

# DeepGP
sbatch slurm/test_al_gp_deep.sh

# TabPFN
sbatch slurm/test_al_tabpfn.sh
