import json
import tempfile
import unittest
from pathlib import Path

from polymer_network import load_project_config


class ParamsTests(unittest.TestCase):
    def test_load_project_config_reads_network_simulation_and_run_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(
                json.dumps(
                    {
                        "output": {"prefix": "case_a"},
                        "network": {
                            "cells_x": 3,
                            "cells_y": 2,
                            "insertion_density": 0.75,
                            "horizontal_ratio": 0.25,
                            "seed": 77,
                        },
                        "simulation": {
                            "tstar_list": ["0.40", "1.00"],
                            "production_time_lj": "50",
                            "pre_relax_steps": 25,
                        },
                        "run": {
                            "mpi_ranks": 4,
                            "omp_threads": 1,
                            "result_dir": "Result_case_a",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_project_config(path)

        self.assertEqual(config.output.prefix, "case_a")
        self.assertEqual(config.network.cells_x, 3)
        self.assertEqual(config.network.cells_y, 2)
        self.assertEqual(config.network.insertion_density, 0.75)
        self.assertEqual(config.network.horizontal_ratio, 0.25)
        self.assertEqual(config.network.seed, 77)
        self.assertEqual(config.simulation.tstar_list, ("0.40", "1.00"))
        self.assertEqual(config.simulation.production_time_lj, "50")
        self.assertEqual(config.simulation.pre_relax_steps, 25)
        self.assertEqual(config.run.mpi_ranks, 4)
        self.assertEqual(config.run.omp_threads, 1)
        self.assertEqual(config.run.result_dir, "Result_case_a")

    def test_load_project_config_reads_3d_cubic_network_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(
                json.dumps(
                    {
                        "network": {
                            "topology_mode": "unified_lattice",
                            "cells_x": 2,
                            "cells_y": 3,
                            "cells_z": 4,
                            "x_s_count": 5,
                            "y_s_count": 7,
                            "z_s_count": 9,
                            "insertion_density": 0.5,
                            "axis_insertion_weights": {"x": 2, "y": 3, "z": 5},
                            "seed": 42,
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_project_config(path)

        self.assertEqual(config.network.topology_mode, "unified_lattice")
        self.assertEqual(config.network.cells_x, 2)
        self.assertEqual(config.network.cells_y, 3)
        self.assertEqual(config.network.cells_z, 4)
        self.assertEqual(config.network.resolved_x_s_count, 5)
        self.assertEqual(config.network.resolved_y_s_count, 7)
        self.assertEqual(config.network.resolved_z_s_count, 9)
        self.assertEqual(
            config.network.effective_axis_insertion_weights,
            {"x": 2.0, "y": 3.0, "z": 5.0},
        )
        self.assertEqual(
            config.network.normalized_axis_insertion_ratios,
            {"x": 0.2, "y": 0.3, "z": 0.5},
        )

    def test_load_project_config_reads_cooldown_simulation_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(
                json.dumps(
                    {
                        "simulation": {
                            "temperature_protocol": "cooldown",
                            "tstar_list": ["1.40", "1.20", "1.00"],
                            "output_tag": "lcden125",
                            "cooldown_ramp_time_lj": "2",
                            "cooldown_relax_time_lj": "3",
                            "cooldown_sample_time_lj": "8",
                            "ramp_dump_frames": 2,
                            "relax_dump_frames": 3,
                            "sample_dump_frames": 8,
                            "volume_sample_every": 25,
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_project_config(path)

        self.assertEqual(config.simulation.temperature_protocol, "cooldown")
        self.assertEqual(config.simulation.tstar_list, ("1.40", "1.20", "1.00"))
        self.assertEqual(config.simulation.output_tag, "lcden125")
        self.assertEqual(config.simulation.cooldown_ramp_time_lj, "2")
        self.assertEqual(config.simulation.cooldown_relax_time_lj, "3")
        self.assertEqual(config.simulation.cooldown_sample_time_lj, "8")
        self.assertEqual(config.simulation.ramp_dump_frames, 2)
        self.assertEqual(config.simulation.relax_dump_frames, 3)
        self.assertEqual(config.simulation.sample_dump_frames, 8)
        self.assertEqual(config.simulation.volume_sample_every, 25)

    def test_load_project_config_maps_independent_protocol_alias_to_temperature_sweep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "params.json"
            path.write_text(
                json.dumps({"simulation": {"temperature_protocol": "independent"}}),
                encoding="utf-8",
            )

            config = load_project_config(path)

        self.assertEqual(config.simulation.temperature_protocol, "temperature_sweep")


if __name__ == "__main__":
    unittest.main()
