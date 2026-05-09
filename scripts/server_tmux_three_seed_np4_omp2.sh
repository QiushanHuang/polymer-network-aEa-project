#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEEDS="${SEEDS:-12345 22345 32345}"
export SUITE_DIR="${SUITE_DIR:-experiments/volume_phase_np4_omp2_3seed_tmux}"
export CPU_TOTAL="${CPU_TOTAL:-256}"

exec "$SCRIPT_DIR/server_tmux_two_seed_np4_omp2.sh"
