#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from polymer_network import load_project_config
from polymer_network.runner import run_lammps_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate files from params.json and run LAMMPS with the configured MPI/OMP settings."
    )
    parser.add_argument("--params", default="params.json", help="Path to the JSON parameter file.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Root used to resolve relative output/result paths. Default: repository root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate files and print the command without launching LAMMPS.")
    parser.add_argument("--skip-generate", action="store_true", help="Use existing generated .in/.data files.")
    parser.add_argument("--mpi-ranks", type=int, default=None, help="Override run.mpi_ranks.")
    parser.add_argument("--omp-threads", type=int, default=None, help="Override run.omp_threads.")
    parser.add_argument("--mpiexec", default=None, help="Override run.mpiexec.")
    parser.add_argument("--lammps-bin", default=None, help="Override run.lammps_bin.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else CODE_ROOT
    params_path = _resolve_from_root(args.params, project_root)
    config = _with_run_overrides(load_project_config(params_path), args)
    return run_lammps_case(
        config,
        project_root=project_root,
        dry_run=args.dry_run,
        skip_generate=args.skip_generate,
    )


def _with_run_overrides(config, args: argparse.Namespace):
    run = config.run
    if args.mpi_ranks is not None:
        run = replace(run, mpi_ranks=args.mpi_ranks)
    if args.omp_threads is not None:
        run = replace(run, omp_threads=args.omp_threads)
    if args.mpiexec is not None:
        run = replace(run, mpiexec=args.mpiexec)
    if args.lammps_bin is not None:
        run = replace(run, lammps_bin=args.lammps_bin)
    return replace(config, run=run)


def _resolve_from_root(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


if __name__ == "__main__":
    raise SystemExit(main())
