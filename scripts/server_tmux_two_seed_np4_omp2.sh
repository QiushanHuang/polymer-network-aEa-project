#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$(pwd)}"

PARAMS="${PARAMS:-$CODE_ROOT/params_volume_phase.json}"
SUITE_DIR="${SUITE_DIR:-experiments/volume_phase_np4_omp2_2seed_tmux}"

MPI_RANKS="${MPI_RANKS:-4}"
OMP_THREADS="${OMP_THREADS:-2}"
MPIEXEC="${MPIEXEC:-mpiexec}"
LAMMPS_BIN="${LAMMPS_BIN:-lmp_mpi}"

CPU_TOTAL="${CPU_TOTAL:-256}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
WAIT_FOR_FINISH="${WAIT_FOR_FINISH:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"

SESSION_PREFIX="${SESSION_PREFIX:-lce_np4omp2}"
ANALYSIS_METHOD="${ANALYSIS_METHOD:-gaussian}"
ANALYSIS_GRID_SPACING="${ANALYSIS_GRID_SPACING:-0.5}"
ANALYSIS_BINS="${ANALYSIS_BINS:-50}"
ANALYSIS_STRIDE="${ANALYSIS_STRIDE:-1}"

SEEDS="${SEEDS:-12345 22345}"
DENSITIES="${DENSITIES:-0.000 0.125 0.250 0.375 0.500 0.625 0.750}"

read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a DENSITY_ARGS <<< "$DENSITIES"

CASE_COUNT=$(( ${#SEED_ARGS[@]} * ${#DENSITY_ARGS[@]} ))
CPUS_PER_CASE=$(( MPI_RANKS * OMP_THREADS ))
REQUESTED_CPUS=$(( CASE_COUNT * CPUS_PER_CASE ))

if ! command -v tmux >/dev/null 2>&1; then
  echo "Cannot find tmux. Install tmux or load the server module first." >&2
  exit 2
fi

if (( CASE_COUNT <= 0 )); then
  echo "No cases requested. Check SEEDS and DENSITIES." >&2
  exit 2
fi

if (( CPUS_PER_CASE <= 0 )); then
  echo "MPI_RANKS and OMP_THREADS must be positive." >&2
  exit 2
fi

if (( REQUESTED_CPUS > CPU_TOTAL )); then
  echo "This tmux launcher starts one tmux session per case." >&2
  echo "Requested CPUs: ${REQUESTED_CPUS}; CPU_TOTAL=${CPU_TOTAL}." >&2
  echo "Reduce DENSITIES/SEEDS, or raise CPU_TOTAL if your allocation is larger." >&2
  exit 2
fi

MANIFEST="$RUN_ROOT/$SUITE_DIR/manifest.json"
TMUX_DIR="$RUN_ROOT/$SUITE_DIR/tmux"
RUNNER_DIR="$TMUX_DIR/runners"
LOG_DIR="$TMUX_DIR/logs"
STATUS_DIR="$TMUX_DIR/status"
mkdir -p "$RUNNER_DIR" "$LOG_DIR" "$STATUS_DIR"

echo "Code root    : $CODE_ROOT"
echo "Run root     : $RUN_ROOT"
echo "Params       : $PARAMS"
echo "Suite        : $RUN_ROOT/$SUITE_DIR"
echo "Seeds        : ${SEED_ARGS[*]}"
echo "Densities    : ${DENSITY_ARGS[*]}"
echo "Cases        : $CASE_COUNT"
echo "Per case     : np=${MPI_RANKS}, omp=${OMP_THREADS}, CPUs=${CPUS_PER_CASE}"
echo "tmux sessions: $CASE_COUNT"
echo "CPU request  : ${REQUESTED_CPUS} / ${CPU_TOTAL}"
echo "MPI command  : ${MPIEXEC} -np ${MPI_RANKS} ${LAMMPS_BIN}"

PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python3 "$CODE_ROOT/scripts/create_volume_phase_experiments.py" \
  --project-root "$RUN_ROOT" \
  --params "$PARAMS" \
  --suite-dir "$SUITE_DIR" \
  --densities "${DENSITY_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}"

PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python3 "$CODE_ROOT/scripts/run_experiment_matrix.py" \
  --project-root "$RUN_ROOT" \
  --manifest "$SUITE_DIR/manifest.json" \
  --mpi-ranks "$MPI_RANKS" \
  --omp-threads "$OMP_THREADS" \
  --mpiexec "$MPIEXEC" \
  --lammps-bin "$LAMMPS_BIN" \
  --jobs "$CASE_COUNT" \
  --dry-run

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
  echo "DRY_RUN_ONLY=1, stopping after input generation and command validation."
  exit 0
fi

CASE_LIST="$TMUX_DIR/cases.tsv"
python3 - "$MANIFEST" <<'PY' > "$CASE_LIST"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in manifest.get("cases", []):
    print(case["case_id"], case["params_path"], case["result_dir"], sep="\t")
PY

SESSION_LIST="$TMUX_DIR/tmux_sessions.txt"
: > "$SESSION_LIST"

while IFS=$'\t' read -r case_id params_path result_dir; do
  session="${SESSION_PREFIX}_${case_id}"
  params_abs="$params_path"
  result_abs="$result_dir"
  [[ "$params_abs" = /* ]] || params_abs="$RUN_ROOT/$params_abs"
  [[ "$result_abs" = /* ]] || result_abs="$RUN_ROOT/$result_abs"

  runner_script="$RUNNER_DIR/${case_id}.sh"
  tmux_log="$LOG_DIR/${case_id}.tmux.log"
  status_file="$STATUS_DIR/${case_id}.status"
  mkdir -p "$result_abs"

  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE_EXISTING" == "1" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session" >&2
      echo "Set REPLACE_EXISTING=1 to kill and relaunch it." >&2
      exit 2
    fi
  fi

  cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -u
cd $(printf '%q' "$RUN_ROOT")
export PYTHONPATH=$(printf '%q' "$CODE_ROOT"):\${PYTHONPATH:-}
echo "[START] $case_id \$(date)"
echo "[CASE] $case_id" > $(printf '%q' "$tmux_log")
echo "[RUN_ROOT] $RUN_ROOT" >> $(printf '%q' "$tmux_log")
echo "[RESULT] $result_abs" >> $(printf '%q' "$tmux_log")
echo "[COMMAND] OMP_NUM_THREADS=$OMP_THREADS $MPIEXEC -np $MPI_RANKS $LAMMPS_BIN" >> $(printf '%q' "$tmux_log")
set +e
python3 $(printf '%q' "$CODE_ROOT/scripts/run_lammps_from_params.py") \\
  --project-root $(printf '%q' "$RUN_ROOT") \\
  --params $(printf '%q' "$params_abs") \\
  --mpi-ranks $MPI_RANKS \\
  --omp-threads $OMP_THREADS \\
  --mpiexec $(printf '%q' "$MPIEXEC") \\
  --lammps-bin $(printf '%q' "$LAMMPS_BIN") \\
  --skip-generate \\
  >> $(printf '%q' "$tmux_log") 2>&1
rc=\$?
set -e
echo "\$rc" > $(printf '%q' "$status_file")
if [[ "\$rc" -eq 0 ]]; then
  echo "[DONE] $case_id \$(date)" >> $(printf '%q' "$tmux_log")
else
  echo "[FAILED] $case_id rc=\$rc \$(date)" >> $(printf '%q' "$tmux_log")
fi
exit "\$rc"
EOF
  chmod +x "$runner_script"

  rm -f "$status_file"
  tmux new-session -d -s "$session" "bash $(printf '%q' "$runner_script")"
  printf "%s\t%s\t%s\t%s\n" "$session" "$case_id" "$result_abs" "$tmux_log" >> "$SESSION_LIST"
  echo "Launched $session -> $result_abs"
done < "$CASE_LIST"

CHECK_SCRIPT="$TMUX_DIR/check_tmux_status.sh"
cat > "$CHECK_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "Active tmux sessions:"
tmux list-sessions -F '#S' 2>/dev/null | grep '^$(printf '%s' "$SESSION_PREFIX" | sed 's/[][\.^$*+?{}|()]/\\&/g')_' || true
echo
echo "Case status files:"
for status in $(printf '%q' "$STATUS_DIR")/*.status; do
  [[ -e "\$status" ]] || continue
  printf "%s " "\$(basename "\$status" .status)"
  cat "\$status"
done
EOF
chmod +x "$CHECK_SCRIPT"

echo
echo "tmux sessions written to: $SESSION_LIST"
echo "tmux logs are under     : $LOG_DIR"
echo "case outputs are under  : $RUN_ROOT/$SUITE_DIR/cases/<case_id>/"
echo "Check status with       : $CHECK_SCRIPT"
FIRST_SESSION="$(awk 'NR == 1 {print $1}' "$SESSION_LIST")"
echo "Attach example          : tmux attach -t $FIRST_SESSION"

if [[ "$WAIT_FOR_FINISH" == "1" ]]; then
  echo "WAIT_FOR_FINISH=1, waiting for all tmux sessions to exit..."
  while true; do
    active=0
    while IFS=$'\t' read -r session _case_id _result_dir _log_path; do
      if tmux has-session -t "$session" 2>/dev/null; then
        active=$((active + 1))
      fi
    done < "$SESSION_LIST"
    if (( active == 0 )); then
      break
    fi
    echo "Active sessions: $active"
    sleep 60
  done

  failures=0
  while IFS=$'\t' read -r _session case_id _result_dir _log_path; do
    status_file="$STATUS_DIR/${case_id}.status"
    if [[ ! -s "$status_file" ]] || [[ "$(cat "$status_file")" != "0" ]]; then
      echo "Failed or missing status: $case_id" >&2
      failures=$((failures + 1))
    fi
  done < "$SESSION_LIST"
  if (( failures > 0 )); then
    exit 1
  fi

  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python3 "$CODE_ROOT/scripts/analyze_volume_probability.py" \
      "$RUN_ROOT/$SUITE_DIR" \
      --method "$ANALYSIS_METHOD" \
      --grid-spacing "$ANALYSIS_GRID_SPACING" \
      --bins "$ANALYSIS_BINS" \
      --stride "$ANALYSIS_STRIDE"
  fi
fi
