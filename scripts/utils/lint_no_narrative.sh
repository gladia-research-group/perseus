#!/bin/bash
# lint_no_narrative.sh — fail if development-log narrative leaked into the published tree:
# ledger names, campaign dates, machine and user names, job ids. Run by CI.
cd "$(dirname "$0")/../.."
PATTERN='KNOWLEDGE\.md|TO-TRY|FAILURE\.md|SUCCESSFUL\.md|SETUPS\.md|CLAUDE\.md|ROADMAP|\badd\.[0-9]{2,3}[a-z]?\b|2026-[0-9]{2}-[0-9]{2}|behemoth|leonardo|sbatch|SLURM|azirilli|mithrillo|user-1004|IscrC_|EUHPC_|USER-SIGNED|⚠|⛔'
if grep -rniE "$PATTERN" --include='*.py' --include='*.cu' --include='*.cpp' --include='*.h' --include='*.cuh' \
     --include='*.sh' --include='*.md' --include='*.toml' --include='*.yml' --include='*.yaml' --include='*.ipynb' --include='*.json' \
     --include='CMakeLists.txt' --include='.gitignore' --include='.gitmodules' \
     perseus src include scripts tests notebooks docs configs .github .gitignore .gitmodules \
     third_party/openfhe-n32 \
     README.md CONTRIBUTING.md CMakeLists.txt pyproject.toml \
     2>/dev/null | grep -vE 'scripts/utils/lint_no_narrative.sh|tests/test_notebooks.py:[0-9]+:FORBIDDEN' \
     | grep .; then
    echo "narrative lint: the lines above must not ship"; exit 1
fi
echo "narrative lint: clean"
