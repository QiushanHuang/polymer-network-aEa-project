import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from polymer_network.analysis.dump import parse_lammps_dump_frames
from polymer_network.analysis.volume import (
    convex_hull_volume,
    ellipsoid_metric_q2,
    gaussian_density_volume,
)
from polymer_network.analysis.volume_probability import analyze_volume_probability
from polymer_network.analysis.volume_probability_v2 import (
    FRAME_CACHE_DAT,
    analyze_volume_probability_v2,
    build_arg_parser,
    load_frame_cache,
    resolve_input_mode,
    run_analysis,
)


def _dump_text(include_unwrapped=True, timestep=100):
    coord_cols = "x y z xu yu zu" if include_unwrapped else "x y z"
    atom_rows = [
        "1 1 1 1 1 1 1 1 1 0 0 0 1 1 3 1.0",
        "2 2 2 1 1 2 1 1 1 0 0 0 1 1 1 0.333333",
        "3 3 8 8 8 8 8 8 1 0 0 0 0.00001 0.00001 0.00001 0.00001",
    ]
    if not include_unwrapped:
        atom_rows = [
            "1 1 1 1 1 1 0 0 0 1 1 3 1.0",
            "2 2 2 1 1 1 0 0 0 1 1 1 0.333333",
            "3 3 8 8 8 1 0 0 0 0.00001 0.00001 0.00001 0.00001",
        ]
    return "\n".join(
        [
            "ITEM: TIMESTEP",
            str(timestep),
            "ITEM: NUMBER OF ATOMS",
            "3",
            "ITEM: BOX BOUNDS pp pp pp",
            "0 10",
            "0 10",
            "0 10",
            f"ITEM: ATOMS id type {coord_cols} quatw quati quatj quatk shapex shapey shapez mass",
            *atom_rows,
            "",
        ]
    )


class VolumeAnalysisTests(unittest.TestCase):
    def test_parse_dump_requires_unwrapped_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.dump"
            path.write_text(_dump_text(include_unwrapped=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "xu yu zu"):
                list(parse_lammps_dump_frames(path, require_unwrapped=True))

    def test_gaussian_density_volume_uses_selected_types_and_ignores_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "good.dump"
            path.write_text(_dump_text(include_unwrapped=True), encoding="utf-8")
            frame = next(parse_lammps_dump_frames(path))

        volume = gaussian_density_volume(
            frame.atoms,
            grid_spacing=1.0,
            threshold=0.60,
            selected_types={1, 2},
        )

        self.assertGreater(volume, 0.0)
        self.assertTrue(all(atom.atom_type != 3 for atom in frame.selected_atoms({1, 2})))

    def test_gaussian_density_volume_sums_overlapping_particle_density(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "good.dump"
            path.write_text(_dump_text(include_unwrapped=True), encoding="utf-8")
            frame = next(parse_lammps_dump_frames(path))
        atom = frame.atoms[1]
        single = gaussian_density_volume(
            [atom],
            grid_spacing=1.0,
            threshold=1.10,
            selected_types={2},
        )
        doubled = gaussian_density_volume(
            [atom, atom],
            grid_spacing=1.0,
            threshold=1.10,
            selected_types={2},
        )

        self.assertEqual(single, 0.0)
        self.assertGreater(doubled, 0.0)

    def test_gaussian_density_volume_is_translation_invariant_for_grid_origin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "good.dump"
            path.write_text(_dump_text(include_unwrapped=True), encoding="utf-8")
            frame = next(parse_lammps_dump_frames(path))
        atoms = frame.selected_atoms({1, 2})
        shifted = tuple(
            replace(
                atom,
                x=atom.x + 0.13,
                y=atom.y + 0.17,
                z=atom.z + 0.19,
                xu=atom.xu + 0.13,
                yu=atom.yu + 0.17,
                zu=atom.zu + 0.19,
            )
            for atom in atoms
        )

        original = gaussian_density_volume(atoms, grid_spacing=0.5, threshold=0.60)
        translated = gaussian_density_volume(shifted, grid_spacing=0.5, threshold=0.60)

        self.assertEqual(original, translated)

    def test_ellipsoid_metric_uses_quaternion_orientation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "good.dump"
            path.write_text(_dump_text(include_unwrapped=True), encoding="utf-8")
            frame = next(parse_lammps_dump_frames(path))
        e_atom = frame.atoms[0]
        sqrt_half = 0.5 ** 0.5
        x_oriented = type(e_atom)(
            atom_id=e_atom.atom_id,
            atom_type=e_atom.atom_type,
            x=e_atom.x,
            y=e_atom.y,
            z=e_atom.z,
            xu=e_atom.xu,
            yu=e_atom.yu,
            zu=e_atom.zu,
            quat=(sqrt_half, 0.0, sqrt_half, 0.0),
            shape=e_atom.shape,
            mass=e_atom.mass,
        )

        self.assertGreater(ellipsoid_metric_q2(e_atom, e_atom.xu + 1.4, e_atom.yu, e_atom.zu), 1.0)
        self.assertLess(ellipsoid_metric_q2(x_oriented, e_atom.xu + 1.4, e_atom.yu, e_atom.zu), 1.0)

    def test_analyze_volume_probability_reads_sample_dumps_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tstar_dir = root / "cases" / "lcden125_seed12345" / "result" / "Tstar_1.40"
            tstar_dir.mkdir(parents=True)
            (tstar_dir / "HEAV.lcden125.ramp.0.dump").write_text(_dump_text(True), encoding="utf-8")
            (tstar_dir / "HEAV.lcden125.relax.0.dump").write_text(_dump_text(True), encoding="utf-8")
            (tstar_dir / "HEAV.lcden125.sample.0.dump").write_text(_dump_text(True), encoding="utf-8")
            out_dir = root / "analysis"

            result = analyze_volume_probability(
                root,
                out_dir=out_dir,
                method="gaussian",
                bins=4,
                grid_spacing=1.0,
                threshold=0.60,
                block_size=1,
                tail_frames_per_tstar=1,
                stride=1,
            )

            self.assertEqual(len(result.frame_records), 1)
            self.assertEqual(result.frame_records[0]["phase"], "sample")
            summary_text = (out_dir / "Volume_probability_summary.dat").read_text(encoding="utf-8")
            self.assertIn("# method = gaussian", summary_text)
            self.assertIn("# grid_spacing = 1", summary_text)
            self.assertIn("# frame_selection = sample", summary_text)
            self.assertIn("lcden125_seed12345", summary_text)
            bins_text = (out_dir / "Volume_probability_bins_all_cases.dat").read_text(encoding="utf-8")
            self.assertIn("probability_density", bins_text)
            phase_text = (out_dir / "Volume_phase_diagram.dat").read_text(encoding="utf-8")
            self.assertIn("lcden_tag", phase_text)
            self.assertIn("seed_count", phase_text)
            self.assertIn("lcden125", phase_text)

    def test_analyze_volume_probability_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No sample dump frames"):
                analyze_volume_probability(Path(temp_dir), out_dir=Path(temp_dir) / "analysis")

    def test_volume_probability_v2_writes_and_reuses_frame_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tstar_dir = root / "cases" / "lcden125_seed12345" / "result" / "Tstar_1.40"
            tstar_dir.mkdir(parents=True)
            for index in range(3):
                (tstar_dir / f"HEAV.lcden125.sample.{index}.dump").write_text(
                    _dump_text(True, timestep=100 + index),
                    encoding="utf-8",
                )
            out_dir = root / "analysis_v2"

            result = analyze_volume_probability_v2(
                root,
                out_dir=out_dir,
                input_mode="dump",
                bins=4,
                grid_spacing=1.0,
                threshold=0.60,
                block_size=1,
                tail_frames_per_tstar=1,
                write_plots=False,
            )

            cache_path = out_dir / FRAME_CACHE_DAT
            self.assertTrue(cache_path.exists())
            cache_text = cache_path.read_text(encoding="utf-8")
            self.assertIn("row_kind", cache_text)
            self.assertIn("volume", cache_text)
            self.assertEqual(len(result.frame_records), 1)
            self.assertTrue((out_dir / "Volume_probability_summary_V2.dat").exists())
            self.assertTrue((out_dir / "Volume_probability_bins_all_cases_V2.dat").exists())

            cached = load_frame_cache(cache_path)
            self.assertEqual(len(cached), 3)
            self.assertEqual(cached[0]["case_label"], "lcden125_seed12345")
            self.assertEqual(cached[0]["tstar"], "Tstar_1.40")
            self.assertEqual(float(cached[-1]["volume"]), float(result.frame_records[0]["volume"]))

            self.assertEqual(resolve_input_mode("auto", root, out_dir=out_dir), "dat")
            self.assertEqual(resolve_input_mode("auto", root, out_dir=out_dir, method="convex_hull"), "dump")
            with self.assertRaisesRegex(ValueError, "requested method=convex_hull"):
                load_frame_cache(cache_path, required_method="convex_hull")
            for dump_path in tstar_dir.glob("*.dump"):
                dump_path.unlink()

            dat_result = analyze_volume_probability_v2(
                root,
                out_dir=out_dir,
                input_mode="dat",
                bins=4,
                grid_spacing=1.0,
                threshold=0.60,
                block_size=1,
                min_timestep=101,
                max_timestep=102,
                write_plots=False,
            )

            self.assertEqual([row["timestep"] for row in dat_result.frame_records], [101, 102])

    def test_volume_probability_v2_cli_defaults_output_to_root_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tstar_dir = root / "cases" / "lcden125_seed12345" / "result" / "Tstar_1.40"
            tstar_dir.mkdir(parents=True)
            (tstar_dir / "HEAV.lcden125.sample.0.dump").write_text(_dump_text(True), encoding="utf-8")
            args = build_arg_parser().parse_args(
                [
                    str(root),
                    "--no-plots",
                    "--bins",
                    "4",
                    "--grid-spacing",
                    "1.0",
                    "--threshold",
                    "0.60",
                    "--block-size",
                    "1",
                    "--quiet",
                ]
            )

            result = run_analysis(args)

            self.assertEqual(result.out_dir, root.resolve())
            self.assertTrue((root / FRAME_CACHE_DAT).exists())
            self.assertTrue((root / "Volume_probability_summary_V2.dat").exists())

    def test_volume_probability_v2_cli_reports_dump_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tstar_dir = root / "cases" / "lcden125_seed12345" / "result" / "Tstar_1.40"
            tstar_dir.mkdir(parents=True)
            for index in range(2):
                (tstar_dir / f"HEAV.lcden125.sample.{index}.dump").write_text(
                    _dump_text(True, timestep=100 + index),
                    encoding="utf-8",
                )
            args = build_arg_parser().parse_args(
                [
                    str(root),
                    "--no-plots",
                    "--input-mode",
                    "dump",
                    "--bins",
                    "4",
                    "--grid-spacing",
                    "1.0",
                    "--threshold",
                    "0.60",
                    "--block-size",
                    "1",
                    "--progress-every",
                    "1",
                ]
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_analysis(args)

            text = output.getvalue()
            self.assertIn("[INFO] root =", text)
            self.assertIn("found 2 sample dump files", text)
            self.assertIn("processing dump 1/2", text)
            self.assertIn("processed 2/2 dump files", text)
            self.assertIn("wrote frame cache", text)

    def test_convex_hull_reports_missing_scipy(self):
        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

        try:
            import scipy  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(RuntimeError, "SciPy"):
                convex_hull_volume(points)
        else:
            self.assertGreater(convex_hull_volume(points), 0.0)


if __name__ == "__main__":
    unittest.main()
