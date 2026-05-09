import json
import tempfile
import unittest
from pathlib import Path

from polymer_network import NetworkConfig, ProjectConfig, SimulationConfig
from polymer_network.experiments import (
    build_volume_phase_cases,
    lc_insertion_fraction_tag,
    write_volume_phase_suite,
)


class ExperimentMatrixTests(unittest.TestCase):
    def test_lc_insertion_fraction_tag_uses_per_mille_precision(self):
        self.assertEqual(lc_insertion_fraction_tag(0.0), "lcden000")
        self.assertEqual(lc_insertion_fraction_tag(0.125), "lcden125")
        self.assertEqual(lc_insertion_fraction_tag(0.375), "lcden375")
        self.assertEqual(lc_insertion_fraction_tag("0.750"), "lcden750")

    def test_build_volume_phase_cases_creates_isolated_cooldown_cases(self):
        base = ProjectConfig(
            network=NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=7,
                cells_y=7,
                cells_z=7,
                x_s_count=10,
                y_s_count=10,
                z_s_count=10,
                insertion_density=0.25,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                contact_gap=0.01,
                seed=12345,
            ),
            simulation=SimulationConfig(
                tstar_list=("1.40", "1.35", "1.30"),
                cooldown_ramp_time_lj="1",
                cooldown_relax_time_lj="1",
                cooldown_sample_time_lj="2",
                ramp_dump_frames=1,
                relax_dump_frames=1,
                sample_dump_frames=2,
            ),
        )

        cases = build_volume_phase_cases(
            base,
            suite_dir="experiments/test_suite",
            densities=(0.0, 0.125, 0.25),
            seeds=(12345, 22345),
        )

        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case.case_id for case in cases}), 6)
        self.assertEqual(len({case.config.output.output_dir for case in cases}), 6)
        self.assertEqual(len({case.config.run.result_dir for case in cases}), 6)
        case = next(item for item in cases if item.lc_insertion_fraction == 0.125)
        self.assertEqual(case.lcden_tag, "lcden125")
        self.assertEqual(case.config.network.insertion_density, 0.125)
        self.assertEqual(case.config.network.axis_insertion_weights, {"x": 1, "y": 1, "z": 1})
        self.assertEqual(case.config.simulation.temperature_protocol, "cooldown")
        self.assertEqual(case.config.simulation.output_tag, case.lcden_tag)
        self.assertEqual(case.manifest_record["maps_to"], "network.insertion_density")
        self.assertEqual(case.manifest_record["axis_insertions"], {"x": 56, "y": 56, "z": 56})
        self.assertEqual(case.manifest_record["actual_insertion_count"], 168)
        self.assertAlmostEqual(case.manifest_record["actual_lc_insertion_fraction"], 0.125)

    def test_build_volume_phase_cases_reverses_default_ascending_schedule_for_cooldown(self):
        base = ProjectConfig(
            network=NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=7,
                cells_y=7,
                cells_z=7,
                x_s_count=10,
                y_s_count=10,
                z_s_count=10,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                contact_gap=0.01,
            )
        )

        cases = build_volume_phase_cases(
            base,
            suite_dir="experiments/test_suite",
            densities=(0.125,),
            seeds=(12345,),
        )

        self.assertEqual(cases[0].config.simulation.tstar_list[0], "1.40")
        self.assertEqual(cases[0].config.simulation.tstar_list[-1], "0.50")
        self.assertEqual(cases[0].manifest_record["temperature_schedule"][0], "1.40")

    def test_build_volume_phase_cases_rejects_non_3d_xyz_equal_matrix(self):
        base = ProjectConfig(
            network=NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=7,
                cells_y=7,
                cells_z=1,
                axis_insertion_weights={"x": 1, "y": 1, "z": 0},
            )
        )

        with self.assertRaisesRegex(ValueError, "cells_z > 1"):
            build_volume_phase_cases(base, suite_dir="experiments/test_suite", densities=(0.125,), seeds=(1,))

    def test_write_volume_phase_suite_writes_manifest_and_case_params(self):
        base = ProjectConfig(
            network=NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=7,
                cells_y=7,
                cells_z=7,
                x_s_count=10,
                y_s_count=10,
                z_s_count=10,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=12345,
            ),
            simulation=SimulationConfig(tstar_list=("1.40", "1.35")),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_dir = Path(temp_dir) / "suite"

            manifest_path = write_volume_phase_suite(
                base,
                project_root=temp_dir,
                suite_dir=suite_dir,
                densities=(0.125,),
                seeds=(12345,),
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["suite_id"], "suite")
            self.assertEqual(manifest["temperature_schedule"], ["1.40", "1.35"])
            self.assertEqual(len(manifest["cases"]), 1)
            params_path = Path(temp_dir) / manifest["cases"][0]["params_path"]
            self.assertTrue(params_path.exists())
            params = json.loads(params_path.read_text(encoding="utf-8"))
            self.assertEqual(params["network"]["insertion_density"], 0.125)
            self.assertEqual(params["simulation"]["temperature_protocol"], "cooldown")
            self.assertEqual(params["simulation"]["output_tag"], "lcden125")
            self.assertEqual(params["simulation"]["tstar_list"], ["1.40", "1.35"])


if __name__ == "__main__":
    unittest.main()
