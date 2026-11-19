#!/bin/bash
# Fetch the two estimators the scripts run on, at the commits every result here was computed with, into security/ext/.
#   ext/lattice-estimator   malb/lattice-estimator 6019056 (standard attacks, Ogilvie, the scans)
#   ext/est_hj              the estimator Hou-Jiang forked (lattice-estimator e35f45b) with their lwe_primal.py and
#                           cost.py from identitymapping/enhanced_lattice-estimator f8968ad9 (nothing else of theirs)
#   ext/enhanced_lattice-estimator   their repository itself (FHE_paper_estimate_results.csv = their Table 5.1)
# Both upstreams are LGPLv3+ (see their READMEs); nothing of them is copied into this repository.
set -euo pipefail
cd "$(dirname "$0")"
LE_URL=https://github.com/malb/lattice-estimator.git
LE_COMMIT=6019056011d10d7e9c30a0d5da2d2f729fbc2eec
LE_HJ_BASE=e35f45b7976a90a79c3c6625a45bbc344c1abc67
HJ_URL=https://github.com/identitymapping/enhanced_lattice-estimator.git
HJ_COMMIT=f8968ad9cba1d6fe0aeaa825d7ad6e39ea3551a3
mkdir -p ext
cd ext

if [ ! -d lattice-estimator ]; then
    git clone -q "$LE_URL" lattice-estimator
fi
git -C lattice-estimator checkout -q "$LE_COMMIT"

if [ ! -d enhanced_lattice-estimator ]; then
    git clone -q "$HJ_URL" enhanced_lattice-estimator
fi
git -C enhanced_lattice-estimator checkout -q "$HJ_COMMIT"

rm -rf est_hj
mkdir est_hj
git -C lattice-estimator archive "$LE_HJ_BASE" estimator | tar -x -C est_hj
cp enhanced_lattice-estimator/estimator/lwe_primal.py enhanced_lattice-estimator/estimator/cost.py est_hj/estimator/

echo "lattice-estimator   $(git -C lattice-estimator rev-parse HEAD)"
echo "enhanced estimator  $(git -C enhanced_lattice-estimator rev-parse HEAD)"
echo "est_hj              lattice-estimator $LE_HJ_BASE + their lwe_primal.py, cost.py"
