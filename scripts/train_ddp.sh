#!/usr/bin/env bash
# Thin alias: multi-GPU DDP via the unified train.sh entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/train.sh" "$@"
