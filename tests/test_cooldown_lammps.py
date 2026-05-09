import unittest

from polymer_network import NetworkConfig, SimulationConfig, build_polymer_network
from polymer_network.lammps import render_lammps_input


class CooldownLammpsTests(unittest.TestCase):
    def test_cooldown_protocol_outputs_high_temperature_ramp_relax_sample_then_cooling_stages(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                x_s_count=7,
                y_s_count=7,
                z_s_count=7,
                insertion_density=0.25,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=5,
            )
        )

        text = render_lammps_input(
            network,
            data_filename="network.data",
            simulation=SimulationConfig(
                temperature_protocol="cooldown",
                tstar_list=("1.40", "1.20", "1.00"),
                output_tag="lcden250",
                timestep="0.001",
                cooldown_ramp_time_lj="2",
                cooldown_relax_time_lj="3",
                cooldown_sample_time_lj="4",
                ramp_dump_frames=2,
                relax_dump_frames=3,
                sample_dump_frames=4,
                restart_interval_dump_frames=2,
                velocity_seed=24680,
            ),
        )

        self.assertIn("COOLDOWN PROTOCOL", text)
        self.assertEqual(text.count("read_data       network.data"), 1)
        self.assertEqual(text.count("velocity        all create"), 1)
        self.assertNotIn("label           loop_Tstar", text)
        self.assertIn("shell           mkdir -p Tstar_1.40", text)
        self.assertIn("Tstar_1.40/HEAV.lcden250.ramp.*.dump", text)
        self.assertIn("Tstar_1.40/HEAV.lcden250.relax.*.dump", text)
        self.assertIn("Tstar_1.40/HEAV.lcden250.sample.*.dump", text)
        self.assertIn("Tstar_1.20/HEAV.lcden250.ramp.*.dump", text)
        self.assertIn("Tstar_1.20/HEAV.lcden250.relax.*.dump", text)
        self.assertIn("Tstar_1.20/HEAV.lcden250.sample.*.dump", text)
        self.assertIn("Tstar_1.00/Restart.cooldown.lcden250.*", text)
        self.assertIn("Tstar_1.00/Final.cooldown.lcden250.bin", text)
        self.assertIn("file Tstar_1.20/volume_state.lcden250.sample.dat", text)
        self.assertIn("variable        DTRamp", text)
        self.assertIn("variable        DTRelax", text)
        self.assertIn("variable        DTSample", text)
        self.assertIn("fix             temp_control_lc rigid_lc rigid/nvt/small molecule", text)

    def test_cooldown_zero_insertion_uses_sphere_only_thermostat(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                insertion_density=0.0,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=5,
            )
        )

        text = render_lammps_input(
            network,
            data_filename="network.data",
            simulation=SimulationConfig(
                temperature_protocol="cooldown",
                tstar_list=("1.40", "1.20"),
                output_tag="lcden000",
                cooldown_ramp_time_lj="1",
                cooldown_relax_time_lj="1",
                cooldown_sample_time_lj="1",
                ramp_dump_frames=1,
                relax_dump_frames=1,
                sample_dump_frames=1,
            ),
        )

        self.assertEqual(network.count_atoms(1), 0)
        self.assertNotIn("rigid_lc rigid/nvt/small molecule", text)
        self.assertIn("fix             temp_control_sph sphere nvt", text)
        self.assertIn("variable        T_ell     equal 0.0", text)

    def test_cooldown_rejects_non_integer_cadence(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                insertion_density=0.0,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=5,
            )
        )

        with self.assertRaisesRegex(ValueError, "SampleSteps/sample_dump_frames"):
            render_lammps_input(
                network,
                data_filename="network.data",
                simulation=SimulationConfig(
                    temperature_protocol="cooldown",
                    tstar_list=("1.40", "1.20"),
                    timestep="0.001",
                    cooldown_ramp_time_lj="1",
                    cooldown_relax_time_lj="1",
                    cooldown_sample_time_lj="1",
                    ramp_dump_frames=1,
                    relax_dump_frames=1,
                    sample_dump_frames=3,
                ),
            )


if __name__ == "__main__":
    unittest.main()
