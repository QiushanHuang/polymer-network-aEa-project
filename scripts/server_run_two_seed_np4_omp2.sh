#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PARAMS="${PARAMS:-params_volume_phase.json}"
SUITE_DIR="${SUITE_DIR:-experiments/volume_phase_np4_omp2_2seed}"

MPI_RANKS="${MPI_RANKS:-4}"
OMP_THREADS="${OMP_THREADS:-2}"
MPIEXEC="${MPIEXEC:-mpiexec}"
LAMMPS_BIN="${LAMMPS_BIN:-lmp_mpi}"

CPU_TOTAL="${CPU_TOTAL:-256}"
JOBS="${JOBS:-14}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
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
MAX_JOBS_BY_CPU=$(( CPU_TOTAL / CPUS_PER_CASE ))

if (( CASE_COUNT <= 0 )); then
  echo "No cases requested. Check SEEDS and DENSITIES." >&2
  exit 2
fi

if (( MAX_JOBS_BY_CPU <= 0 )); then
  echo "CPU_TOTAL=${CPU_TOTAL} is smaller than one case cost: ${CPUS_PER_CASE} CPUs." >&2
  exit 2
fi

if (( JOBS > MAX_JOBS_BY_CPU )); then
  echo "JOBS=${JOBS} would request $(( JOBS * CPUS_PER_CASE )) CPUs, above CPU_TOTAL=${CPU_TOTAL}." >&2
  echo "Use JOBS<=${MAX_JOBS_BY_CPU}, or increase CPU_TOTAL if your allocation is larger." >&2
  exit 2
fi

if (( JOBS > CASE_COUNT )); then
  JOBS="$CASE_COUNT"
fi

echo "Project      : $ROOT_DIR"
echo "Params       : $PARAMS"
echo "Suite        : $SUITE_DIR"
echo "Seeds        : ${SEED_ARGS[*]}"
echo "Densities    : ${DENSITY_ARGS[*]}"
echo "Cases        : $CASE_COUNT"
echo "Per case     : np=${MPI_RANKS}, omp=${OMP_THREADS}, CPUs=${CPUS_PER_CASE}"
echo "Parallel jobs: $JOBS"
echo "CPU request  : $(( JOBS * CPUS_PER_CASE )) / ${CPU_TOTAL}"
echo "MPI command  : ${MPIEXEC} -np ${MPI_RANKS} ${LAMMPS_BIN}"

python3 scripts/create_volume_phase_experiments.py \
  --params "$PARAMS" \
  --suite-dir "$SUITE_DIR" \
  --densities "${DENSITY_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}"

python3 scripts/run_experiment_matrix.py \
  --manifest "$SUITE_DIR/manifest.json" \
  --mpi-ranks "$MPI_RANKS" \
  --omp-threads "$OMP_THREADS" \
  --mpiexec "$MPIEXEC" \
  --lammps-bin "$LAMMPS_BIN" \
  --jobs "$JOBS" \
  --dry-run

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
  echo "DRY_RUN_ONLY=1, stopping after input generation and command validation."
  exit 0
fi

python3 scripts/run_experiment_matrix.py \
  --manifest "$SUITE_DIR/manifest.json" \
  --mpi-ranks "$MPI_RANKS" \
  --omp-threads "$OMP_THREADS" \
  --mpiexec "$MPIEXEC" \
  --lammps-bin "$LAMMPS_BIN" \
  --jobs "$JOBS"

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  python3 scripts/analyze_volume_probability.py \
    "$SUITE_DIR" \
    --method "$ANALYSIS_METHOD" \
    --grid-spacing "$ANALYSIS_GRID_SPACING" \
    --bins "$ANALYSIS_BINS" \
    --stride "$ANALYSIS_STRIDE"
fi

echo "Done. Suite: $SUITE_DIR"
