#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polymer_network import NetworkConfig, build_polymer_network, load_project_config
from polymer_network.lammps import (
    render_lammps_data,
    render_lammps_input,
    render_metadata_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LAMMPS input/data files for an S polymer network with embedded a-E-a fragments."
    )
    parser.add_argument("--params", default="params.json", help="Path to the JSON parameter file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--topology-mode", choices=("unified_lattice", "2D_surface", "3D_cubic"), default=None)
    parser.add_argument("--cells-x", type=int, default=None)
    parser.add_argument("--cells-y", type=int, default=None)
    parser.add_argument("--cells-z", type=int, default=None)
    parser.add_argument("--horizontal-s-count", type=int, default=None)
    parser.add_argument("--vertical-s-count", type=int, default=None)
    parser.add_argument("--x-s-count", type=int, default=None)
    parser.add_argument("--y-s-count", type=int, default=None)
    parser.add_argument("--z-s-count", type=int, default=None)
    parser.add_argument("--insertion-density", type=float, default=None)
    parser.add_argument("--horizontal-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contact-gap", type=float, default=None)
    parser.add_argument("--box-padding", type=float, default=None)
    parser.add_argument(
        "--relative-data-path",
        action="store_true",
        default=None,
        help="Use a relative read_data path in the .in file. Default uses an absolute path, matching run_all scripts that execute from Result_* directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = load_project_config(args.params)
    output_config = project_config.output
    base_network = project_config.network
    output_dir = Path(_pick(args.output_dir, output_config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = replace(
        base_network,
        topology_mode=_pick(args.topology_mode, base_network.topology_mode),
        cells_x=_pick(args.cells_x, base_network.cells_x),
        cells_y=_pick(args.cells_y, base_network.cells_y),
        cells_z=_pick(args.cells_z, base_network.cells_z),
        horizontal_s_count=_pick(args.horizontal_s_count, base_network.horizontal_s_count),
        vertical_s_count=_pick(args.vertical_s_count, base_network.vertical_s_count),
        x_s_count=_pick(args.x_s_count, base_network.x_s_count),
        y_s_count=_pick(args.y_s_count, base_network.y_s_count),
        z_s_count=_pick(args.z_s_count, base_network.z_s_count),
        insertion_density=_pick(args.insertion_density, base_network.insertion_density),
        horizontal_ratio=_pick(args.horizontal_ratio, base_network.horizontal_ratio),
        seed=_pick(args.seed, base_network.seed),
        contact_gap=_pick(args.contact_gap, base_network.contact_gap),
        box_padding=_pick(args.box_padding, base_network.box_padding),
    )
    network = build_polymer_network(config)

    prefix = _pick(args.prefix, output_config.prefix)
    data_path = output_dir / f"{prefix}.data"
    input_path = output_dir / f"{prefix}.in"
    metadata_path = output_dir / f"{prefix}.metadata.json"
    data_path.write_text(render_lammps_data(network), encoding="utf-8")

    relative_data_path = _pick(args.relative_data_path, output_config.relative_data_path)
    data_reference = data_path.name if relative_data_path else str(data_path.resolve())
    input_path.write_text(
        render_lammps_input(
            network,
            data_filename=data_reference,
            simulation=project_config.simulation,
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(render_metadata_json(network), encoding="utf-8")

    print(f"Wrote {input_path}")
    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")
    print(
        "Counts: "
        f"S={network.count_atoms(2)}, E={network.count_atoms(1)}, "
        f"anchor={network.count_atoms(3)}, bonds={len(network.bonds)}, "
        f"insertions={len(network.insertions)} "
        f"(X={network.count_insertions('x')}, Y={network.count_insertions('y')}, "
        f"Z={network.count_insertions('z')})"
    )
    return 0


def _pick(value, fallback):
    return fallback if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
