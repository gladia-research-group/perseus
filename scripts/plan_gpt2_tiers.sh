#!/bin/bash
# plan_gpt2_tiers.sh — replan the three GPT-2 decode-tier plans (bare smart cut,
# same recipe as make_plans.sh planned_gpt2_base) + refresh the t128 probe copy.
# Run after the scope-fix recaptures land (plans are binary-bound).
set -e -o pipefail
cd "$(dirname "$0")/.."
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export PYTHON="${PYTHON:-/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/.venv/bin/python}"

for tier in base heat squeeze; do
    echo "== plan .cache/graph_gpt2_$tier -> planned_gpt2_$tier"
    env GRAPH_DIR=.cache/graph_gpt2_$tier OUT_NAME=planned_gpt2_$tier \
        bash scripts/utils/run_bootstrap_all_blocks.sh
    echo "   placements: $("$PYTHON" - planned_gpt2_$tier <<'EOF'
import json,glob,sys
print(sum(len(json.load(open(f)).get('placements',[])) for f in glob.glob(f'bootstrap_placements/{sys.argv[1]}/block_*_placement.json')))
EOF
)"
done

rm -rf bootstrap_placements/planned_gpt2_base_t128
cp -r bootstrap_placements/planned_gpt2_base bootstrap_placements/planned_gpt2_base_t128
echo "== t128 probe plan refreshed from planned_gpt2_base"
