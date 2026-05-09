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
from polymer_network.experiments import (
    DEFAULT_LC_INSERTION_FRACTIONS,
    DEFAULT_TOPOLOGY_SEEDS,
    write_volume_phase_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a density x seed cooldown experiment suite for volume phase-diagram runs."
    )
    parser.add_argument("--params", default="params.json", help="Base ProjectConfig JSON.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Root where generated suite files are written. Default: repository root.",
    )
    parser.add_argument("--suite-dir", default="experiments/volume_phase_diagram")
    parser.add_argument("--densities", nargs="*", default=list(DEFAULT_LC_INSERTION_FRACTIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_TOPOLOGY_SEEDS))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small smoke matrix: densities 0.00/0.50 and one seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else CODE_ROOT
    params_path = _resolve_from_root(args.params, project_root)
    config = load_project_config(params_path)
    if args.smoke:
        config = replace(
            config,
            network=replace(
                config.network,
                topology_mode="unified_lattice",
                cells_x=2,
                cells_y=2,
                cells_z=2,
                x_s_count=7,
                y_s_count=7,
                z_s_count=7,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                contact_gap=0.01,
            ),
            simulation=replace(
                config.simulation,
                tstar_list=("1.40", "1.20"),
                cooldown_ramp_time_lj="1",
                cooldown_relax_time_lj="1",
                cooldown_sample_time_lj="2",
                ramp_dump_frames=1,
                relax_dump_frames=1,
                sample_dump_frames=2,
                restart_interval_dump_frames=1,
            ),
        )
    densities = ("0.000", "0.500") if args.smoke else tuple(args.densities)
    seeds = (int(args.seeds[0]),) if args.smoke else tuple(args.seeds)
    manifest = write_volume_phase_suite(
        config,
        project_root=project_root,
        suite_dir=args.suite_dir,
        densities=densities,
        seeds=seeds,
    )
    print(f"Wrote {manifest}")
    return 0


def _resolve_from_root(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


if __name__ == "__main__":
    raise SystemExit(main())
