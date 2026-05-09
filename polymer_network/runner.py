from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .core import build_polymer_network
from .lammps import render_lammps_data, render_lammps_input, render_metadata_json
from .params import ProjectConfig


@dataclass(frozen=True)
class GeneratedPaths:
    output_dir: Path
    data_path: Path
    input_path: Path
    metadata_path: Path
    result_dir: Path
    command: tuple[str, ...]


def prepare_lammps_case(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    skip_generate: bool = False,
) -> GeneratedPaths:
    root = Path(project_root).resolve()
    output_dir = root / config.output.output_dir
    prefix = config.output.prefix
    data_path = output_dir / f"{prefix}.data"
    input_path = output_dir / f"{prefix}.in"
    metadata_path = output_dir / f"{prefix}.metadata.json"

    if not skip_generate:
        output_dir.mkdir(parents=True, exist_ok=True)
        network = build_polymer_network(config.network)
        data_path.write_text(render_lammps_data(network), encoding="utf-8")
        data_reference = data_path.name if config.output.relative_data_path else str(data_path.resolve())
        input_path.write_text(
            render_lammps_input(
                network,
                data_filename=data_reference,
                simulation=config.simulation,
            ),
            encoding="utf-8",
        )
        metadata_path.write_text(render_metadata_json(network), encoding="utf-8")

    result_dir = root / config.run.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    command = (
        config.run.mpiexec,
        "-np",
        str(config.run.mpi_ranks),
        config.run.lammps_bin,
        "-in",
        str(input_path.resolve()),
    )
    command_text = f"OMP_NUM_THREADS={config.run.omp_threads} " + " ".join(command)
    (result_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    return GeneratedPaths(
        output_dir=output_dir,
        data_path=data_path,
        input_path=input_path,
        metadata_path=metadata_path,
        result_dir=result_dir,
        command=command,
    )


def run_lammps_case(
    config: ProjectConfig,
    project_root: str | Path,
    *,
    dry_run: bool = False,
    skip_generate: bool = False,
    stream_output: bool = True,
    verbose: bool = True,
) -> int:
    paths = prepare_lammps_case(
        config,
        project_root=project_root,
        skip_generate=skip_generate,
    )
    if verbose:
        print(f"Input : {paths.input_path}")
        print(f"Data  : {paths.data_path}")
        print(f"Result: {paths.result_dir}")
        print("Command:")
        print(" ".join(paths.command))

    if dry_run:
        return 0

    _validate_runtime(config.run.mpiexec, config.run.lammps_bin)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.run.omp_threads)

    log_path = paths.result_dir / "runner.log"
    if not stream_output:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                list(paths.command),
                cwd=paths.result_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return process.returncode

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(paths.command),
            cwd=paths.result_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def _validate_runtime(mpiexec: str, lammps_bin: str) -> None:
    missing = [binary for binary in (mpiexec, lammps_bin) if shutil.which(binary) is None]
    if missing:
        raise SystemExit(f"Cannot find required executable(s): {', '.join(missing)}")
