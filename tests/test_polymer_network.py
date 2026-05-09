import math
import unittest

from polymer_network import NetworkConfig, build_polymer_network


class PolymerNetworkTests(unittest.TestCase):
    def test_builds_rectangular_s_network_without_insertions(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="2D_surface",
                cells_x=2,
                cells_y=1,
                horizontal_s_count=7,
                vertical_s_count=7,
                insertion_density=0.0,
                seed=123,
            )
        )

        self.assertEqual(network.count_atoms(atom_type=2), 41)
        self.assertEqual(network.count_atoms(atom_type=1), 0)
        self.assertEqual(network.count_atoms(atom_type=3), 0)
        self.assertEqual(len(network.bonds), 42)
        self.assertTrue(all(bond.bond_type == 1 for bond in network.bonds))
        self.assertEqual(network.metadata["topology_mode"], "2D_surface")
        self.assertEqual({site.z for site in network.sites.values()}, {10.0})

    def test_unified_lattice_cells_z_one_is_single_layer_network(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=1,
                insertion_density=0.0,
                seed=123,
            )
        )

        self.assertEqual(network.metadata["topology_mode"], "unified_lattice")
        self.assertEqual(network.count_atoms(atom_type=2), 24)
        self.assertEqual(len(network.bonds), 24)
        self.assertEqual(len(network.segments), 4)
        self.assertEqual(network.metadata["segment_counts_by_axis"], {"x": 2, "y": 2, "z": 0})
        self.assertEqual(len({site.z for site in network.sites.values()}), 1)

    def test_unified_lattice_3d_cells_are_box_counts_on_all_axes(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=3,
                cells_y=4,
                cells_z=5,
                insertion_density=0.0,
                seed=123,
            )
        )

        cross_sites = [network.sites[site_id] for site_id in network.cross_site_ids]
        x_cross = {site.ix for site in cross_sites}
        y_cross = {site.iy for site in cross_sites}
        z_cross = {site.iz for site in cross_sites}

        self.assertEqual((len(x_cross) - 1, len(y_cross) - 1, len(z_cross) - 1), (3, 4, 5))
        self.assertEqual(network.config.z_grid_layers, 6)
        self.assertEqual(network.config.z_cell_count, 5)

    def test_unified_lattice_cells_z_one_rejects_z_insertion_weight(self):
        with self.assertRaisesRegex(ValueError, "z insertion weight"):
            build_polymer_network(
                NetworkConfig(
                    topology_mode="unified_lattice",
                    cells_x=1,
                    cells_y=1,
                    cells_z=1,
                    insertion_density=0.25,
                    axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                    seed=123,
                )
            )

    def test_2d_surface_rejects_z_weight_ratio_and_depth_alias(self):
        cases = [
            {"axis_insertion_weights": {"x": 1, "y": 1, "z": 1}},
            {"axis_insertion_ratios": {"x": 0.4, "y": 0.4, "z": 0.2}},
            {"axis_insertion_weights": {"horizontal": 1, "vertical": 1, "depth": 1}},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "z insertion weight"):
                    build_polymer_network(
                        NetworkConfig(
                            topology_mode="2D_surface",
                            cells_x=1,
                            cells_y=1,
                            cells_z=1,
                            insertion_density=0.25,
                            seed=123,
                            **kwargs,
                        )
                    )

    def test_insertion_density_half_tie_rounds_up(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=2,
                cells_y=2,
                cells_z=2,
                insertion_density=4.5 / 54,
                axis_insertion_weights={"x": 1, "y": 0, "z": 0},
                seed=12,
            )
        )

        self.assertEqual(len(network.insertions), 5)
        self.assertEqual(network.count_insertions("x"), 5)
        self.assertEqual(network.count_insertions("y"), 0)
        self.assertEqual(network.count_insertions("z"), 0)

    def test_insertions_are_reproducible_and_follow_orientation_ratio(self):
        config = NetworkConfig(
            cells_x=2,
            cells_y=1,
            horizontal_s_count=7,
            vertical_s_count=7,
            insertion_density=0.5,
            horizontal_ratio=0.5,
            seed=99,
        )

        first = build_polymer_network(config)
        second = build_polymer_network(config)

        self.assertEqual(
            [(ins.segment_id, ins.start_index, ins.orientation) for ins in first.insertions],
            [(ins.segment_id, ins.start_index, ins.orientation) for ins in second.insertions],
        )
        self.assertEqual(len(first.insertions), 4)
        self.assertEqual(first.count_insertions("horizontal"), 2)
        self.assertEqual(first.count_insertions("vertical"), 2)
        self.assertEqual(first.count_insertions("x"), 2)
        self.assertEqual(first.count_insertions("y"), 2)
        self.assertEqual(first.count_atoms(atom_type=1), 4)
        self.assertEqual(first.count_atoms(atom_type=3), 8)

    def test_insertions_never_replace_cross_nodes_or_overlap(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=3,
                cells_y=2,
                horizontal_s_count=7,
                vertical_s_count=7,
                insertion_density=1.0,
                horizontal_ratio=0.5,
                seed=7,
            )
        )

        replaced = []
        for insertion in network.insertions:
            replaced.extend(insertion.replaced_site_ids)
            self.assertTrue(insertion.replaced_site_ids.isdisjoint(network.cross_site_ids))

        self.assertEqual(len(replaced), len(set(replaced)))

    def test_geometry_records_ss_and_se_distances(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=1,
                cells_y=1,
                horizontal_s_count=7,
                vertical_s_count=7,
                insertion_density=1.0,
                horizontal_ratio=1.0,
                seed=1,
            )
        )

        uninserted_ss_distances = [
            network.distance(bond.atom1, bond.atom2)
            for bond in network.bonds
            if bond.bond_type == 1
        ]
        self.assertTrue(uninserted_ss_distances)
        for distance in uninserted_ss_distances:
            self.assertAlmostEqual(distance, 1.1, places=7)

        for insertion in network.insertions:
            self.assertAlmostEqual(insertion.target_se_distance, 2.1, places=7)
            self.assertAlmostEqual(insertion.actual_left_se_distance, 2.2, places=7)
            self.assertAlmostEqual(insertion.actual_right_se_distance, 2.2, places=7)
            self.assertAlmostEqual(network.distance(insertion.left_neighbor_atom_id, insertion.left_anchor_atom_id), 0.7)
            self.assertAlmostEqual(network.distance(insertion.right_anchor_atom_id, insertion.right_neighbor_atom_id), 0.7)
            self.assertAlmostEqual(network.distance(insertion.left_anchor_atom_id, insertion.e_atom_id), 1.5)
            self.assertAlmostEqual(network.distance(insertion.e_atom_id, insertion.right_anchor_atom_id), 1.5)

    def test_horizontal_and_vertical_quaternions_are_normalized(self):
        network = build_polymer_network(
            NetworkConfig(
                cells_x=1,
                cells_y=1,
                insertion_density=1.0,
                horizontal_ratio=0.5,
                seed=11,
            )
        )

        ellipsoids = [atom for atom in network.atoms if atom.atom_type == 1]
        self.assertTrue(ellipsoids)
        for atom in ellipsoids:
            norm = math.sqrt(sum(component * component for component in atom.quaternion))
            self.assertAlmostEqual(norm, 1.0, places=7)

    def test_3d_cubic_network_without_insertions_builds_cube_edge_lattice(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                x_s_count=7,
                y_s_count=7,
                z_s_count=7,
                insertion_density=0.0,
                seed=123,
            )
        )

        self.assertEqual(network.metadata["topology_mode"], "unified_lattice")
        self.assertEqual(network.count_atoms(atom_type=2), 112)
        self.assertEqual(network.count_atoms(atom_type=1), 0)
        self.assertEqual(network.count_atoms(atom_type=3), 0)
        self.assertEqual(len(network.bonds), 120)
        self.assertEqual(len(network.segments), 20)
        self.assertEqual(len(network.cross_site_ids), 12)
        self.assertGreater(len({site.z for site in network.sites.values()}), 2)

    def test_3d_cubic_z_insertions_use_z_axis_and_preserve_junctions(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=1,
                cells_y=1,
                cells_z=2,
                x_s_count=7,
                y_s_count=7,
                z_s_count=7,
                insertion_density=1.0,
                axis_insertion_weights={"x": 0, "y": 0, "z": 1},
                seed=3,
            )
        )

        self.assertEqual(len(network.insertions), 8)
        self.assertEqual(network.count_insertions("z"), 8)
        for insertion in network.insertions:
            self.assertEqual(insertion.orientation, "z")
            self.assertTrue(insertion.replaced_site_ids.isdisjoint(network.cross_site_ids))
            left_site = network.sites[insertion.left_neighbor_site_id]
            right_site = network.sites[insertion.right_neighbor_site_id]
            e_atom = network.atom_by_id(insertion.e_atom_id)
            self.assertAlmostEqual(left_site.x, right_site.x)
            self.assertAlmostEqual(left_site.y, right_site.y)
            self.assertLess(left_site.z, e_atom.z)
            self.assertLess(e_atom.z, right_site.z)
            self.assertEqual(e_atom.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_3d_cubic_axis_ratios_use_largest_remainder_allocation(self):
        network = build_polymer_network(
            NetworkConfig(
                topology_mode="unified_lattice",
                cells_x=2,
                cells_y=2,
                cells_z=3,
                insertion_density=14 / 75,
                axis_insertion_weights={"x": 1, "y": 1, "z": 1},
                seed=12345,
            )
        )

        self.assertEqual(len(network.insertions), 14)
        self.assertEqual(network.count_insertions("x"), 5)
        self.assertEqual(network.count_insertions("y"), 5)
        self.assertEqual(network.count_insertions("z"), 4)


if __name__ == "__main__":
    unittest.main()
