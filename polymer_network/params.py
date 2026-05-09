from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import NetworkConfig


DEFAULT_TSTAR_LIST = (
    "0.50",
    "0.55",
    "0.60",
    "0.65",
    "0.70",
    "0.75",
    "0.80",
    "0.85",
    "0.90",
    "0.95",
    "1.00",
    "1.05",
    "1.10",
    "1.15",
    "1.20",
    "1.25",
    "1.30",
    "1.35",
    "1.40",
)


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "polymer-network-inputs"
    prefix: str = "MD_polymer_network_aEa"
    relative_data_path: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    tstar_list: tuple[str, ...] = DEFAULT_TSTAR_LIST
    temperature_protocol: str = "temperature_sweep"
    output_tag: str | None = None
    rho: str = "0.3"
    rho_str: str = "0.30"
    rho_tag: str = "rho030"
    heating_t: str = "4.7"
    cooling_t: str = "3.7"
    timestep: str = "0.001"
    production_time_lj: str = "4000"
    cooldown_pre_hold_time_lj: str = "0"
    cooldown_ramp_time_lj: str = "100"
    cooldown_hold_time_lj: str = "4000"
    cooldown_relax_time_lj: str = "300"
    cooldown_sample_time_lj: str = "3600"
    dump_frames: int = 4000
    ramp_dump_frames: int = 100
    relax_dump_frames: int = 300
    sample_dump_frames: int = 3600
    restart_interval_dump_frames: int = 500
    volume_sample_every: int = 1000
    tdamp_factor: int = 100
    tchain: int = 3
    velocity_seed: int = 13579
    pre_relax_steps: int = 5000
    pre_relax_timestep: str = "0.0002"
    pre_relax_bond_k: str = "50.0"
    bond_k_ss: str = "250.0"
    bond_k_se: str = "250.0"
    r0_ss: str = "1.0"
    r0_se: str = "2.0"
    ds_a: str = "0.50"
    d_anchor: str = "0.00001"
    m_anchor: str = "0.00001"
    r0_sa: str = "0.50"
    r0_ae: str = "1.50"
    gb_gamma: str = "1.0"
    gb_upsilon: str = "3"
    gb_mu: str = "1"
    gb_rcutstar: str = "5"
    epsilon0: str = "1"
    sig0: str = "1"
    eps_a: str = "1"
    eps_b: str = "1"
    eps_c: str = "0.2"
    rdf_bins: int = 200
    rdf_nevery: int = 100
    rdf_nrepeat: int = 10
    rdf_nfreq: int = 1000


@dataclass(frozen=True)
class RunConfig:
    mpi_ranks: int = 4
    omp_threads: int = 1
    mpiexec: str = "mpiexec"
    lammps_bin: str = "lmp_mpi"
    result_dir: str = "Result_MD_polymer_network_aEa"


@dataclass(frozen=True)
class ProjectConfig:
    output: OutputConfig = field(default_factory=OutputConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_project_config(path: str | Path | None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()

    config_path = Path(path)
    if not config_path.exists():
        return ProjectConfig()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    return ProjectConfig(
        output=_output_from_dict(data.get("output", {})),
        network=_network_from_dict(data.get("network", {})),
        simulation=_simulation_from_dict(data.get("simulation", {})),
        run=_run_from_dict(data.get("run", {})),
    )


def _output_from_dict(data: Any) -> OutputConfig:
    if not isinstance(data, dict):
        raise ValueError("output config must be an object")
    defaults = OutputConfig()
    return OutputConfig(
        output_dir=str(data.get("output_dir", defaults.output_dir)),
        prefix=str(data.get("prefix", defaults.prefix)),
        relative_data_path=bool(data.get("relative_data_path", defaults.relative_data_path)),
    )


def _network_from_dict(data: Any) -> NetworkConfig:
    if not isinstance(data, dict):
        raise ValueError("network config must be an object")
    defaults = NetworkConfig()
    topology_mode = str(
        data.get(
            "topology_mode",
            data.get("network_form", data.get("geometry_form", defaults.topology_mode)),
        )
    )
    if topology_mode == "cubic_3d":
        topology_mode = "3D_cubic"
    if topology_mode == "surface_2d":
        topology_mode = "2D_surface"
    if topology_mode in ("unified", "network", "lattice"):
        topology_mode = "unified_lattice"

    axis_weights = data.get(
        "axis_insertion_weights",
        data.get("direction_weights", defaults.axis_insertion_weights),
    )
    axis_ratios = data.get(
        "axis_insertion_ratios",
        data.get("direction_ratios", defaults.axis_insertion_ratios),
    )
    return NetworkConfig(
        topology_mode=topology_mode,  # type: ignore[arg-type]
        cells_x=int(data.get("cells_x", defaults.cells_x)),
        cells_y=int(data.get("cells_y", defaults.cells_y)),
        cells_z=int(data.get("cells_z", defaults.cells_z)),
        horizontal_s_count=int(data.get("horizontal_s_count", defaults.horizontal_s_count)),
        vertical_s_count=int(data.get("vertical_s_count", defaults.vertical_s_count)),
        x_s_count=_optional_int(data.get("x_s_count", defaults.x_s_count)),
        y_s_count=_optional_int(data.get("y_s_count", defaults.y_s_count)),
        z_s_count=int(data.get("z_s_count", defaults.z_s_count)),
        insertion_density=float(data.get("insertion_density", defaults.insertion_density)),
        horizontal_ratio=float(data.get("horizontal_ratio", defaults.horizontal_ratio)),
        axis_insertion_weights=_optional_axis_weights(axis_weights),
        axis_insertion_ratios=_optional_axis_ratios(axis_ratios),
        seed=data.get("seed", defaults.seed),
        contact_gap=float(data.get("contact_gap", defaults.contact_gap)),
        box_padding=float(data.get("box_padding", defaults.box_padding)),
        z_plane=_optional_float(data.get("z_plane", defaults.z_plane)),
        s_diameter=float(data.get("s_diameter", defaults.s_diameter)),
        e_long_diameter=float(data.get("e_long_diameter", defaults.e_long_diameter)),
        e_short_diameter=float(data.get("e_short_diameter", defaults.e_short_diameter)),
        anchor_diameter=float(data.get("anchor_diameter", defaults.anchor_diameter)),
        mass_e=float(data.get("mass_e", defaults.mass_e)),
        mass_s=float(data.get("mass_s", defaults.mass_s)),
        mass_anchor=float(data.get("mass_anchor", defaults.mass_anchor)),
        geometry_mode=str(data.get("geometry_mode", defaults.geometry_mode)),  # type: ignore[arg-type]
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_axis_ratios(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("axis_insertion_ratios must be an object")
    return {str(axis): float(weight) for axis, weight in value.items()}


def _optional_axis_weights(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("axis_insertion_weights must be an object")
    return {str(axis): float(weight) for axis, weight in value.items()}


def _simulation_from_dict(data: Any) -> SimulationConfig:
    if not isinstance(data, dict):
        raise ValueError("simulation config must be an object")
    defaults = SimulationConfig()
    tstar_list = data.get("tstar_list", defaults.tstar_list)
    return SimulationConfig(
        tstar_list=tuple(str(value) for value in tstar_list),
        temperature_protocol=_temperature_protocol(data.get("temperature_protocol", defaults.temperature_protocol)),
        output_tag=_optional_str(data.get("output_tag", defaults.output_tag)),
        rho=str(data.get("rho", defaults.rho)),
        rho_str=str(data.get("rho_str", defaults.rho_str)),
        rho_tag=str(data.get("rho_tag", defaults.rho_tag)),
        heating_t=str(data.get("heating_t", defaults.heating_t)),
        cooling_t=str(data.get("cooling_t", defaults.cooling_t)),
        timestep=str(data.get("timestep", defaults.timestep)),
        production_time_lj=str(data.get("production_time_lj", defaults.production_time_lj)),
        cooldown_pre_hold_time_lj=str(data.get("cooldown_pre_hold_time_lj", defaults.cooldown_pre_hold_time_lj)),
        cooldown_ramp_time_lj=str(data.get("cooldown_ramp_time_lj", defaults.cooldown_ramp_time_lj)),
        cooldown_hold_time_lj=str(data.get("cooldown_hold_time_lj", defaults.cooldown_hold_time_lj)),
        cooldown_relax_time_lj=str(data.get("cooldown_relax_time_lj", defaults.cooldown_relax_time_lj)),
        cooldown_sample_time_lj=str(
            data.get("cooldown_sample_time_lj", data.get("cooldown_hold_time_lj", defaults.cooldown_sample_time_lj))
        ),
        dump_frames=int(data.get("dump_frames", defaults.dump_frames)),
        ramp_dump_frames=int(data.get("ramp_dump_frames", defaults.ramp_dump_frames)),
        relax_dump_frames=int(data.get("relax_dump_frames", defaults.relax_dump_frames)),
        sample_dump_frames=int(data.get("sample_dump_frames", data.get("dump_frames", defaults.sample_dump_frames))),
        restart_interval_dump_frames=int(data.get("restart_interval_dump_frames", defaults.restart_interval_dump_frames)),
        volume_sample_every=int(data.get("volume_sample_every", defaults.volume_sample_every)),
        tdamp_factor=int(data.get("tdamp_factor", defaults.tdamp_factor)),
        tchain=int(data.get("tchain", defaults.tchain)),
        velocity_seed=int(data.get("velocity_seed", defaults.velocity_seed)),
        pre_relax_steps=int(data.get("pre_relax_steps", defaults.pre_relax_steps)),
        pre_relax_timestep=str(data.get("pre_relax_timestep", defaults.pre_relax_timestep)),
        pre_relax_bond_k=str(data.get("pre_relax_bond_k", defaults.pre_relax_bond_k)),
        bond_k_ss=str(data.get("bond_k_ss", defaults.bond_k_ss)),
        bond_k_se=str(data.get("bond_k_se", defaults.bond_k_se)),
        r0_ss=str(data.get("r0_ss", defaults.r0_ss)),
        r0_se=str(data.get("r0_se", defaults.r0_se)),
        ds_a=str(data.get("ds_a", defaults.ds_a)),
        d_anchor=str(data.get("d_anchor", defaults.d_anchor)),
        m_anchor=str(data.get("m_anchor", defaults.m_anchor)),
        r0_sa=str(data.get("r0_sa", defaults.r0_sa)),
        r0_ae=str(data.get("r0_ae", defaults.r0_ae)),
        gb_gamma=str(data.get("gb_gamma", defaults.gb_gamma)),
        gb_upsilon=str(data.get("gb_upsilon", defaults.gb_upsilon)),
        gb_mu=str(data.get("gb_mu", defaults.gb_mu)),
        gb_rcutstar=str(data.get("gb_rcutstar", defaults.gb_rcutstar)),
        epsilon0=str(data.get("epsilon0", defaults.epsilon0)),
        sig0=str(data.get("sig0", defaults.sig0)),
        eps_a=str(data.get("eps_a", defaults.eps_a)),
        eps_b=str(data.get("eps_b", defaults.eps_b)),
        eps_c=str(data.get("eps_c", defaults.eps_c)),
        rdf_bins=int(data.get("rdf_bins", defaults.rdf_bins)),
        rdf_nevery=int(data.get("rdf_nevery", defaults.rdf_nevery)),
        rdf_nrepeat=int(data.get("rdf_nrepeat", defaults.rdf_nrepeat)),
        rdf_nfreq=int(data.get("rdf_nfreq", defaults.rdf_nfreq)),
    )


def _temperature_protocol(value: Any) -> str:
    protocol = str(value)
    if protocol == "independent":
        return "temperature_sweep"
    if protocol not in ("temperature_sweep", "cooldown"):
        raise ValueError("temperature_protocol must be 'temperature_sweep' or 'cooldown'")
    return protocol


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _run_from_dict(data: Any) -> RunConfig:
    if not isinstance(data, dict):
        raise ValueError("run config must be an object")
    defaults = RunConfig()
    return RunConfig(
        mpi_ranks=int(data.get("mpi_ranks", defaults.mpi_ranks)),
        omp_threads=int(data.get("omp_threads", defaults.omp_threads)),
        mpiexec=str(data.get("mpiexec", defaults.mpiexec)),
        lammps_bin=str(data.get("lammps_bin", defaults.lammps_bin)),
        result_dir=str(data.get("result_dir", defaults.result_dir)),
    )
