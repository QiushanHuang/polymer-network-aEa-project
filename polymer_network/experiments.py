from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from .core import build_polymer_network
from .params import OutputConfig, ProjectConfig, RunConfig


DEFAULT_LC_INSERTION_FRACTIONS = ("0.000", "0.125", "0.250", "0.375", "0.500", "0.625", "0.750")
DEFAULT_TOPOLOGY_SEEDS = (12345, 22345, 32345)


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    lc_insertion_fraction: float
    liquid_crystal_density: float
    lcden_tag: str
    topology_seed: int
    velocity_seed: int
    config: ProjectConfig
    manifest_record: dict[str, object]


def lc_insertion_fraction_tag(value: float | str | Decimal) -> str:
    fraction = Decimal(str(value))
    if fraction < 0 or fraction > 1:
        raise ValueError("lc insertion fraction must be in [0, 1]")
    scaled = fraction * Decimal(1000)
    if scaled != scaled.to_integral_value():
        raise ValueError("lc insertion fraction tag requires per-mille precision")
    return f"lcden{int(scaled):03d}"


def build_volume_phase_cases(
    base_config: ProjectConfig,
    *,
    suite_dir: str | Path,
    densities: Iterable[float | str] = DEFAULT_LC_INSERTION_FRACTIONS,
    seeds: Iterable[int] = DEFAULT_TOPOLOGY_SEEDS,
    velocity_seed_offset: int = 100000,
    require_equal_axis_counts: bool = True,
) -> list[ExperimentCase]:
    _validate_xyz_equal_base(base_config)
    suite_path = Path(suite_dir)
    temperature_schedule = _cooldown_schedule(base_config.simulation.tstar_list)
    cases: list[ExperimentCase] = []
    for density_value in densities:
        density = float(Decimal(str(density_value)))
        tag = lc_insertion_fraction_tag(density_value)
        for seed in seeds:
            seed = int(seed)
            case_id = f"{tag}_seed{seed}"
            case_root = suite_path / "cases" / case_id
            network_config = replace(
                base_config.network,
                insertion_density=density,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=seed,
            )
            network = build_polymer_network(network_config)
            axis_counts = {
                "x": network.count_insertions("x"),
                "y": network.count_insertions("y"),
                "z": network.count_insertions("z"),
            }
            actual_insertion_count = sum(axis_counts.values())
            eligible_segment_count = int(
                sum(network.metadata.get("segment_counts_by_axis", {}).values())  # type: ignore[union-attr]
            )
            actual_fraction = (
                actual_insertion_count / eligible_segment_count
                if eligible_segment_count > 0
                else 0.0
            )
            if require_equal_axis_counts and len(set(axis_counts.values())) > 1:
                raise ValueError(
                    f"{case_id} does not realize equal xyz insertions: {axis_counts}"
                )

            velocity_seed = seed + velocity_seed_offset
            output_config = OutputConfig(
                output_dir=str(case_root / "inputs"),
                prefix=f"MD_{case_id}",
                relative_data_path=base_config.output.relative_data_path,
            )
            simulation_config = replace(
                base_config.simulation,
                temperature_protocol="cooldown",
                tstar_list=temperature_schedule,
                output_tag=tag,
                velocity_seed=velocity_seed,
            )
            run_config = RunConfig(
                mpi_ranks=base_config.run.mpi_ranks,
                omp_threads=base_config.run.omp_threads,
                mpiexec=base_config.run.mpiexec,
                lammps_bin=base_config.run.lammps_bin,
                result_dir=str(case_root / "result"),
            )
            config = ProjectConfig(
                output=output_config,
                network=network_config,
                simulation=simulation_config,
                run=run_config,
            )
            manifest_record = {
                "case_id": case_id,
                "lc_insertion_fraction": density,
                "liquid_crystal_density": density,
                "target_lc_insertion_fraction": density,
                "actual_lc_insertion_fraction": actual_fraction,
                "eligible_segment_count": eligible_segment_count,
                "target_insertion_count": int(density * eligible_segment_count + 0.5),
                "actual_insertion_count": actual_insertion_count,
                "maps_to": "network.insertion_density",
                "density_basis": "eligible non-junction segment replacement fraction",
                "lcden_tag": tag,
                "params_path": str(case_root / "params.json"),
                "input_dir": str(case_root / "inputs"),
                "result_dir": str(case_root / "result"),
                "network_seed": seed,
                "velocity_seed": velocity_seed,
                "axis_insertion_weights": {"x": 1, "y": 1, "z": 1},
                "axis_insertion_ratios": network.config.normalized_axis_insertion_ratios,
                "axis_insertions": axis_counts,
                "temperature_protocol": simulation_config.temperature_protocol,
                "temperature_schedule": list(temperature_schedule),
            }
            cases.append(
                ExperimentCase(
                    case_id=case_id,
                    lc_insertion_fraction=density,
                    liquid_crystal_density=density,
                    lcden_tag=tag,
                    topology_seed=seed,
                    velocity_seed=velocity_seed,
                    config=config,
                    manifest_record=manifest_record,
                )
            )
    return cases


def write_volume_phase_suite(
    base_config: ProjectConfig,
    *,
    project_root: str | Path,
    suite_dir: str | Path,
    densities: Iterable[float | str] = DEFAULT_LC_INSERTION_FRACTIONS,
    seeds: Iterable[int] = DEFAULT_TOPOLOGY_SEEDS,
) -> Path:
    root = Path(project_root).resolve()
    suite_path = Path(suite_dir)
    if not suite_path.is_absolute():
        suite_path = root / suite_path
    suite_ref = _relative_path(suite_path, root)
    cases = build_volume_phase_cases(
        base_config,
        suite_dir=suite_ref,
        densities=densities,
        seeds=seeds,
    )

    suite_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "polymer-network-aEa-volume-phase-suite-v1",
        "suite_id": suite_path.name,
        "density_basis": "eligible non-junction segment replacement fraction",
        "maps_to": "network.insertion_density",
        "temperature_protocol": "cooldown",
        "temperature_schedule": list(_cooldown_schedule(base_config.simulation.tstar_list)),
        "cases": [],
    }
    for case in cases:
        record = dict(case.manifest_record)
        params_path = root / str(record["params_path"])
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_text(
            json.dumps(asdict(case.config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["cases"].append(record)

    manifest_path = suite_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _validate_xyz_equal_base(config: ProjectConfig) -> None:
    network = config.network
    if not network.has_z_segments:
        raise ValueError("xyz-equal volume phase experiments require cells_z > 1")
    if not (network.cells_x == network.cells_y == network.cells_z):
        raise ValueError("xyz-equal volume phase experiments require cells_x=cells_y=cells_z")
    if not (
        network.resolved_x_s_count
        == network.resolved_y_s_count
        == network.resolved_z_s_count
    ):
        raise ValueError("xyz-equal volume phase experiments require equal x/y/z S counts")


def _relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path


def _cooldown_schedule(values: tuple[str, ...]) -> tuple[str, ...]:
    numeric = [float(value) for value in values]
    if all(left >= right for left, right in zip(numeric, numeric[1:])):
        return values
    if all(left <= right for left, right in zip(numeric, numeric[1:])):
        return tuple(reversed(values))
    raise ValueError("cooldown temperature schedule must be monotonic")
