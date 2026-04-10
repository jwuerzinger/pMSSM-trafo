#!/bin/bash
# Submit all active learning jobs to Slurm.
# Run from repo root: bash submit_slurm.sh

set -euo pipefail

mkdir -p /raven/u/jwuerzin/pMSSM-trafo/logs

# Transformer
sbatch slurm/submit_al_transformer.sh
sbatch slurm/submit_al_transformer_top_k.sh
sbatch slurm/submit_al_transformer_top_k_20k.sh

# ExactGP
sbatch slurm/submit_al_gp_exact.sh
sbatch slurm/submit_al_gp_exact_top_k.sh

# DeepGP
sbatch slurm/submit_al_gp_deep.sh
sbatch slurm/submit_al_gp_deep_top_k.sh

# TabPFN
sbatch slurm/submit_al_tabpfn.sh
sbatch slurm/submit_al_tabpfn_entropy.sh
