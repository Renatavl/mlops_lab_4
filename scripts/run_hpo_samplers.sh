#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
N_TRIALS="${N_TRIALS:-20}"

echo "=== HPO with TPE sampler (n_trials=$N_TRIALS) ==="
python src/optimize.py hpo=tpe hpo.n_trials="$N_TRIALS"

echo ""
echo "=== HPO with Random sampler (n_trials=$N_TRIALS) ==="
python src/optimize.py hpo=random hpo.n_trials="$N_TRIALS"

echo ""
echo "Done. Compare studies in MLflow UI."
