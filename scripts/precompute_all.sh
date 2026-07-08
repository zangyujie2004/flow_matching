bash 

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
  "configs/test3/chahua_all_eef_v2_rel_tac.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do