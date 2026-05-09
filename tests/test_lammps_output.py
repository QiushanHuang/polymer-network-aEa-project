import json
import tempfile
import unittest
from pathlib import Path

from polymer_network import NetworkConfig, SimulationConfig, build_polymer_network
from polymer_network.lammps import render_lammps_data, render_lammps_input, render_metadata_json


class LammpsOutputTests(unittest.TestCase):
    def test_lammps_data_uses_legacy_hybrid_ellipsoid_bond_format(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=1,
                cells_y=1,
                insertion_density=1.0,
                horizontal_ratio=1.0,
                seed=1,
            )
        )

        text = render_lammps_data(network)
        lines = text.splitlines()

        self.assertIn(f"{len(network.atoms)} atoms", lines)
        self.assertIn("3 atom types", lines)
        self.assertIn(f"{len(network.bonds)} bonds", lines)
        self.assertIn("4 bond types", lines)
        self.assertIn(f"{len(network.atoms)} ellipsoids", lines)
        self.assertIn("Atoms # hybrid", lines)
        self.assertIn("Bond Coeffs # harmonic", lines)
        self.assertIn("Ellipsoids", lines)

        atoms_start = lines.index("Atoms # hybrid") + 2
        first_atom_fields = lines[atoms_start].split()
        self.assertEqual(len(first_atom_fields), 11)

        bonds_start = lines.index("Bonds") + 2
        first_bond_fields = lines[bonds_start].split()
        self.assertEqual(len(first_bond_fields), 4)

        ellipsoids_start = lines.index("Ellipsoids") + 2
        first_ellipsoid_fields = lines[ellipsoids_start].split()
        self.assertEqual(len(first_ellipsoid_fields), 8)

        atom_ids = {atom.atom_id for atom in network.atoms}
        for bond in network.bonds:
            self.assertIn(bond.atom1, atom_ids)
            self.assertIn(bond.atom2, atom_ids)

    def test_rendered_input_reads_generated_data_and_preserves_key_legacy_parameters(self):
        network = build_polymer_network(NetworkConfig(cells_x=1, cells_y=1, seed=5))
        text = render_lammps_input(network, data_filename="network.data")

        self.assertIn("units           lj", text)
        self.assertIn("atom_style      hybrid ellipsoid bond", text)
        self.assertIn("read_data       network.data", text)
        self.assertIn("variable        Tstar_list index 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40", text)
        self.assertIn("compute         T_sph sphere temp", text)
        self.assertIn("variable        T_mix", text)
        self.assertIn("pair_style      hybrid gayberne", text)
        self.assertIn("pair_coeff      1 2 gayberne", text)
        self.assertIn("fix             temp_control_lc rigid_lc rigid/nvt/small molecule", text)
        self.assertIn("fix             temp_control_sph sphere nvt", text)
        self.assertIn("write_data      ${Tstar_tag}/TOPOLOGY.${insertion_density_tag}.data", text)
        self.assertIn("variable        Network_topology_mode string unified_lattice", text)
        self.assertIn("variable        Network_cells_z equal 1", text)
        self.assertNotIn("AEA_", text)

    def test_rendered_input_writes_outputs_directly_under_tstar_with_insertion_density_tag(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=1,
                cells_y=1,
                insertion_density=0.25,
                seed=5,
            )
        )
        text = render_lammps_input(network, data_filename="network.data")

        self.assertIn("variable        insertion_density_tag string rho025", text)
        self.assertIn("shell           mkdir -p ${Tstar_tag}", text)
        self.assertNotIn("${Tstar_tag}/${rho_tag}", text)
        self.assertNotIn("variable        rho_tag", text)
        self.assertIn("dump            1 all custom ${DT} ${Tstar_tag}/HEAV.${insertion_density_tag}.*.dump", text)
        self.assertIn("write_data      ${Tstar_tag}/TOPOLOGY.${insertion_density_tag}.data", text)
        self.assertIn("restart         ${DR} ${Tstar_tag}/Restart.GB.${insertion_density_tag}.*", text)
        self.assertIn("file ${Tstar_tag}/rdf.${insertion_density_tag}.dat", text)
        self.assertIn("write_restart   ${Tstar_tag}/Final.${insertion_density_tag}.bin", text)

    def test_rendered_input_quotes_read_data_paths_with_spaces(self):
        network = build_polymer_network(NetworkConfig(cells_x=1, cells_y=1, seed=5))
        text = render_lammps_input(
            network,
            data_filename="/tmp/New project 3/network.data",
        )

        self.assertIn('read_data       "/tmp/New project 3/network.data"', text)

    def test_metadata_marks_none_seed_as_random_mode(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=1,
                cells_y=1,
                insertion_density=0.0,
                seed=None,
            )
        )
        payload = json.loads(render_metadata_json(network))

        self.assertEqual(payload["metadata"]["seed_mode"], "random")
        self.assertIsNone(payload["config"]["seed"])

    def test_zero_insertion_input_guards_ellipsoid_temperature_divide_by_zero(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=1,
                insertion_density=0.0,
                seed=5,
            )
        )
        text = render_lammps_input(network, data_filename="network.data")

        self.assertEqual(network.count_atoms(1), 0)
        self.assertIn("variable        dof_ell   equal 5*count(ellipsoid)", text)
        self.assertIn("variable        T_ell     equal 2.0*(c_KE_ell+c_ER_ell)/(v_dof_ell+1.0e-17)", text)
        self.assertNotIn("variable        T_ell     equal 2.0*(c_KE_ell+c_ER_ell)/v_dof_ell", text)

    def test_rendered_input_uses_simulation_config_values(self):
        network = build_polymer_network(NetworkConfig(cells_x=1, cells_y=1, seed=5))
        text = render_lammps_input(
            network,
            data_filename="network.data",
            simulation=SimulationConfig(
                tstar_list=("0.40", "0.80"),
                timestep="0.002",
                production_time_lj="20",
                dump_frames=100,
                pre_relax_steps=12,
                velocity_seed=24680,
            ),
        )

        self.assertIn("variable        Tstar_list index 0.40 0.80", text)
        self.assertIn("variable        TS      equal 0.002", text)
        self.assertIn("variable        RunTime equal 20/${TS}", text)
        self.assertIn("variable        DT      equal ${RunTime}/100", text)
        self.assertIn("velocity        all create ${Tstar_str} 24680", text)
        self.assertIn("run             12", text)

    def test_rendered_input_exposes_3d_network_shape_without_aEa_bookkeeping(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                insertion_density=0.0,
                axis_insertion_weights={"x": 1, "y": 2, "z": 3},
                seed=5,
            )
        )
        text = render_lammps_input(network, data_filename="network.data")

        self.assertIn("variable        Network_topology_mode string unified_lattice", text)
        self.assertIn("variable        Network_cells_z equal 2", text)
        self.assertIn("variable        X_S_count equal 7", text)
        self.assertIn("variable        Y_S_count equal 7", text)
        self.assertIn("variable        Z_S_count equal 7", text)
        self.assertNotIn("AEA_", text)
        self.assertIn("pair_style      hybrid gayberne", text)
        self.assertIn("fix             temp_control_lc rigid_lc rigid/nvt/small molecule", text)

    def test_generated_files_can_be_written_to_disk(self):
        network = build_polymer_network(NetworkConfig(cells_x=1, cells_y=1, seed=3))
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_path = base / "network.data"
            input_path = base / "network.in"
            data_path.write_text(render_lammps_data(network), encoding="utf-8")
            input_path.write_text(render_lammps_input(network, data_filename=data_path.name), encoding="utf-8")

            self.assertGreater(data_path.stat().st_size, 1000)
            self.assertGreater(input_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
