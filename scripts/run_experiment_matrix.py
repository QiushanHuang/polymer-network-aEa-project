#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from polymer_network import load_project_config, run_lammps_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run every case in a volume phase suite manifest.")
    parser.add_argument("--manifest", required=True, help="Path to experiments/.../manifest.json.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Root used to resolve relative paths in the manifest. Default: repository root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate case inputs and command.txt without LAMMPS.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of cases to run concurrently. Default: 1.")
    parser.add_argument("--mpi-ranks", type=int, default=None, help="Override run.mpi_ranks for every case.")
    parser.add_argument("--omp-threads", type=int, default=None, help="Override run.omp_threads for every case.")
    parser.add_argument("--mpiexec", default=None, help="Override run.mpiexec for every case.")
    parser.add_argument("--lammps-bin", default=None, help="Override run.lammps_bin for every case.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    project_root = Path(args.project_root).resolve() if args.project_root else CODE_ROOT
    manifest_path = _resolve_from_root(args.manifest, project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases", []))
    if args.jobs == 1:
        for case in cases:
            result = _run_case(case, args, project_root, stream_output=True, verbose=True)
            if result.returncode != 0:
                return result.returncode
        return 0

    print(f"Running {len(cases)} cases with jobs={args.jobs}")
    failures: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(_run_case, case, args, project_root, False, False) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            status = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
            print(f"[{status}] {result.case_id} result={result.result_dir} log={result.log_path}")
            if result.returncode != 0:
                failures.append(result)

    if failures:
        print("Failed cases:")
        for result in failures:
            print(f"  {result.case_id}: returncode={result.returncode}, log={result.log_path}")
        return 1
    return 0


class CaseResult:
    def __init__(self, case_id: str, result_dir: Path, log_path: Path, returncode: int) -> None:
        self.case_id = case_id
        self.result_dir = result_dir
        self.log_path = log_path
        self.returncode = returncode


def _run_case(
    case: dict[str, object],
    args: argparse.Namespace,
    project_root: Path,
    stream_output: bool,
    verbose: bool,
) -> CaseResult:
    case_id = str(case["case_id"])
    if verbose:
        print(f"=== {case_id} ===")
    params_path = _resolve_from_root(str(case["params_path"]), project_root)
    config = _with_run_overrides(load_project_config(params_path), args)
    result_dir = _resolve_from_root(config.run.result_dir, project_root)
    code = run_lammps_case(
        config,
        project_root=project_root,
        dry_run=args.dry_run,
        stream_output=stream_output,
        verbose=verbose,
    )
    return CaseResult(case_id, result_dir, result_dir / "runner.log", code)


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
