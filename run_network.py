#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polymer_network import (  # noqa: E402
    NetworkConfig,
    OutputConfig,
    ProjectConfig,
    RunConfig,
    SimulationConfig,
    run_lammps_case,
)


# Edit parameters here, then run:
#   python3 run_network.py
#
# Unified lattice convention:
#   cells_z = 1 means a single z layer, i.e. the 2D surface case.
#   cells_z > 1 adds z-directed segments and becomes a 3D network.
#   insertion_density is the total fraction of eligible network segments selected.
#   axis_insertion_weights splits that total across x:y:z; values are integer-like
#   numeric weights and do not need to sum to 1. Use z=0 when cells_z=1.
#   seed=None requests nondeterministic placement; an integer seed is reproducible.
PROJECT_CONFIG = ProjectConfig(
    output=OutputConfig(
        output_dir="polymer-network-inputs",
        prefix="MD_polymer_network_aEa",
        relative_data_path=False,
    ),
    network=NetworkConfig(
        topology_mode="unified_lattice",
        cells_x=7,
        cells_y=7,
        cells_z=7,
        x_s_count=10,
        y_s_count=10,
        z_s_count=10,
        insertion_density=0.5,
        # Examples:
        #   {"x": 1, "y": 1, "z": 0}  # 2D/surface x:y = 1:1
        #   {"x": 1, "y": 0, "z": 0}  # x-oriented only
        #   {"x": 1, "y": 1, "z": 1}  # 3D isotropic orientation mix
        #   {"x": 8, "y": 1, "z": 1}  # x-biased orientation mix
        axis_insertion_weights={"x": 1, "y": 1, "z": 1},
        seed=12345,
        contact_gap=0.01,
        box_padding=10.0,
        z_plane=None,
        s_diameter=1.0,
        e_long_diameter=3.0,
        e_short_diameter=1.0,
        anchor_diameter=0.00001,
        mass_e=1.0,
        mass_s=1.0 / 3.0,
        mass_anchor=0.00001,
        geometry_mode="preserve-grid",
    ),
    simulation=SimulationConfig(
        tstar_list=(
            # "0.50",
            # "0.55",
            # "0.60",
            # "0.65",
            # "0.70",
            # "0.75",
            # "0.80",
            # "0.85",
            # "0.90",
            # "0.95",
            # "1.00",
            # "1.05",
            # "1.10",
            # "1.15",
            # "1.20",
            # "1.25",
            # "1.30",
            # "1.35",
            "1.40",
        ),
        rho="0.3",
        rho_str="0.30",
        rho_tag="rho030",
        timestep="0.001",
        production_time_lj="4000",
        dump_frames=4000,
        restart_interval_dump_frames=500,
        tdamp_factor=100,
        tchain=3,
        velocity_seed=13579,
        pre_relax_steps=5000,
        pre_relax_timestep="0.0002",
        pre_relax_bond_k="50.0",
        bond_k_ss="250.0",
        bond_k_se="250.0",
        r0_ss="1.0",
        r0_se="2.0",
        ds_a="0.50",
        d_anchor="0.00001",
        m_anchor="0.00001",
        r0_sa="0.50",
        r0_ae="1.50",
        gb_gamma="1.0",
        gb_upsilon="3",
        gb_mu="1",
        gb_rcutstar="5",
        epsilon0="1",
        sig0="1",
        eps_a="1",
        eps_b="1",
        eps_c="0.2",
        rdf_bins=200,
        rdf_nevery=100,
        rdf_nrepeat=10,
        rdf_nfreq=1000,
    ),
    run=RunConfig(
        mpi_ranks=4,
        omp_threads=1,
        mpiexec="mpiexec",
        lammps_bin="lmp_mpi",
        result_dir="Result_MD_polymer_network_aEa",
    ),
)


DRY_RUN = False
SKIP_GENERATE = False


def main() -> int:
    return run_lammps_case(
        PROJECT_CONFIG,
        project_root=PROJECT_ROOT,
        dry_run=DRY_RUN,
        skip_generate=SKIP_GENERATE,
        stream_output=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
