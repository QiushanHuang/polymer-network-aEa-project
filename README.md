# Polymer Network a-E-a Input Generator

This workspace generates LAMMPS inputs for S-particle polymer networks with
embedded `a-E-a` fragments. The normal mode is `unified_lattice`: `cells_z = 1`
is the single-layer surface case, and `cells_z > 1` is the number of z-direction
boxes, matching `cells_x` and `cells_y`. For example, `cells_x=cells_y=cells_z=5`
generates a `5 x 5 x 5` box network with `6 x 6 x 6` junction layers.

The simulation parameters, thermostat strategy, pair coefficients, bond
coefficients, temperature reporting, dump, restart, and RDF output follow the
reference sphere-chain inputs in:

`/Users/joshua/Desktop/MD/2026_04/0408/00_Chain_length/sphere-chain-inputs`

## Direct Run

The simplest workflow is to edit [run_network.py](run_network.py), then run:

```bash
python3 run_network.py
```

That one command regenerates `.in`, `.data`, and `.metadata.json`, creates the
result directory, sets `OMP_NUM_THREADS=1`, launches:

```bash
mpiexec -np 4 lmp_mpi -in /Users/joshua/Desktop/MD/polymer-network-aEa-project/polymer-network-inputs/MD_polymer_network_aEa.in
```

and streams output both to the terminal and to
`Result_MD_polymer_network_aEa/runner.log`.

LAMMPS outputs are written directly under each temperature folder, for example
`Result_MD_polymer_network_aEa/Tstar_1.40/`. There is no nested `rho...`
subfolder. Output filenames use the insertion density tag: `insertion_density=0`
becomes `rho000`, and `insertion_density=0.25` becomes `rho025`.

Do not hand-edit files under `polymer-network-inputs/` for persistent changes.
They are generated outputs and are overwritten whenever generation runs. Change
parameters in `run_network.py`; change LAMMPS template behavior in
`polymer_network/lammps.py`.

Change these values inside [run_network.py](run_network.py):

- `network`: topology mode, cell counts, S count along each axis, a-E-a insertion density, xyz insertion weights, seed, contact gap, particle diameters, and masses.
- `simulation`: LAMMPS temperature list, timestep, production time, dump cadence, restart cadence, bond parameters, GB parameters, RDF output, and velocity seed.
- `run`: MPI ranks, OpenMP threads, LAMMPS executable, and result folder.

The placement controls are:

- `insertion_density`: total fraction of eligible non-junction network segments selected for `a-E-a` insertion.
- `axis_insertion_weights`: raw integer-like numeric xyz direction weights used to split that total insertion target. They are normalized internally and do not need to sum to 1.
- `seed`: random seed for reproducible segment selection and replacement position selection.

The target count is computed as `insertion_density * eligible_segment_count` and
rounded with half-up behavior, so a target of `4.5` becomes `5`. Use an integer
`seed` for reproducible placement. Setting `seed=None` requests nondeterministic
placement; the generated metadata JSON records this as `"seed_mode": "random"`.

For a surface-like case use `cells_z=1` and set the z weight to zero, for
example `axis_insertion_weights={"x": 1, "y": 1, "z": 0}`. For a 3D case use
`cells_z>1`; that value is the z-direction box count. Give z a nonzero weight, for example
`axis_insertion_weights={"x": 1, "y": 1, "z": 1}`. If `cells_z=1` and the z
weight is positive, the generator raises an error instead of silently ignoring
that parameter.

Useful orientation examples:

- `{"x": 1, "y": 0, "z": 0}`: x-oriented insertions only.
- `{"x": 0, "y": 1, "z": 0}`: y-oriented insertions only.
- `{"x": 1, "y": 1, "z": 1}`: equal x/y/z target mix in 3D.
- `{"x": 8, "y": 1, "z": 1}`: x-biased target mix in 3D.

## JSON Workflow

The JSON entry points are still available if you want external parameter files:

```bash
python3 scripts/generate_polymer_network_inputs.py --params params.json
```

Outputs:

- `polymer-network-inputs/MD_polymer_network_aEa.in`
- `polymer-network-inputs/MD_polymer_network_aEa.data`
- `polymer-network-inputs/MD_polymer_network_aEa.metadata.json`

The `.in` file uses an absolute `read_data` path by default so it still works
with the existing `run_all_lammps*.py` style where LAMMPS runs from a separate
`Result_*` directory.

When LAMMPS runs from that result directory, each temperature loop creates
`Tstar_<value>/` and writes files such as `HEAV.rho000.*.dump`,
`TOPOLOGY.rho000.data`, `rdf.rho000.dat`, `Restart.GB.rho000.*`, and
`Final.rho000.bin` directly inside it. The `rho000` part is derived from
`insertion_density`, not from the LAMMPS `rho` variable.

## Model Choices

- Type 1 is `E`, shape `1 1 3`, mass `1`.
- Type 2 is `S`, shape `1 1 1`, mass `1/3`.
- Type 3 is anchor `a`, shape `1e-5 1e-5 1e-5`, mass `1e-5`.
- `a-E-a` is represented by three real atoms with the same molecule id.
- There are no harmonic `a-E`, `E-a`, or `E-S` bonds.
- Existing `S-S` bonds are type 1. New `S-a` bonds are type 2.
- Junction nodes in either topology are never replaced by `a-E-a`.

The generator uses `preserve-grid` geometry. With the requested default gap,
pure `S-S` spacing is `1.0 + 0.1 = 1.1`. Replacing three S particles creates a
fixed-grid neighbor span of `4 * 1.1 = 4.4`, so the initial midpoint `S-E`
distance is `2.2` instead of the target contact value `1.5 + 0.5 + 0.1 = 2.1`.
This is recorded in the metadata JSON for every insertion.

`horizontal_s_count`, `vertical_s_count`, and `horizontal_ratio` are kept for
backward compatibility. New cases should use `x_s_count`, `y_s_count`,
`z_s_count`, `insertion_density`, and `axis_insertion_weights`.

The generated LAMMPS input does not expose `AEA_*` bookkeeping variables because
LAMMPS reads an already-generated `.data` topology. Placement details live in
`MD_polymer_network_aEa.metadata.json`, including raw weights, normalized ratios,
density basis, rounding rule, seed mode, and actual insertion counts by axis.

## Verify

```bash
python3 -m unittest discover -s tests -v
```

## Run

Dry-run the exact command first:

```bash
python3 scripts/run_lammps_from_params.py --params params.json --dry-run
```

`--dry-run` still regenerates the `.in`, `.data`, `.metadata.json`, result
directory, and `command.txt`; it only skips launching LAMMPS.

Launch the configured 4-rank, 1-thread-per-rank run:

```bash
python3 scripts/run_lammps_from_params.py --params params.json
```

The equivalent manual command is:

```bash
cd /Users/joshua/Desktop/MD/polymer-network-aEa-project
mkdir -p Result_MD_polymer_network_aEa
cd Result_MD_polymer_network_aEa
export OMP_NUM_THREADS=1
mpiexec -np 4 lmp_mpi -in ../polymer-network-inputs/MD_polymer_network_aEa.in 2>&1 | tee runner.log
```

Optional LAMMPS smoke check: make a temporary copy of the generated `.in`, keep
only one `Tstar` value, replace both `run` commands with `run 0`, then run
`lmp_mpi -in <smoke.in>`.

## Volume Phase-Diagram Experiments

The batch workflow for volume phase diagrams is separate from the single-case
`run_network.py` path. It treats the user-facing LC density axis as
`network.insertion_density`: the fraction of eligible non-junction network
segments replaced by `a-E-a`. Experiment case names use `lcdenXXX` in
per-mille units, for example `0.125 -> lcden125`. This is not the same as
`simulation.rho`.

Create a quick smoke suite:

```bash
python3 scripts/create_volume_phase_experiments.py --params params.json --smoke --suite-dir experiments/smoke_volume_phase
python3 scripts/run_experiment_matrix.py --manifest experiments/smoke_volume_phase/manifest.json --dry-run
```

For production, start from [params_volume_phase.json](params_volume_phase.json)
or another 3D xyz-equal base config (`cells_z > 1`,
`cells_x=cells_y=cells_z`, equal `x/y/z_s_count`, and
`axis_insertion_weights={"x":1,"y":1,"z":1}`), then omit `--smoke`:

```bash
python3 scripts/create_volume_phase_experiments.py --params params_volume_phase.json --suite-dir experiments/volume_phase_diagram
python3 scripts/run_experiment_matrix.py --manifest experiments/volume_phase_diagram/manifest.json --dry-run
```

The default production matrix is LC insertion fractions `0.000, 0.125, ...,
0.750` across three topology seeds. Each case gets isolated `inputs/`,
`result/`, and `params.json` paths under the suite directory.

The cooldown protocol is enabled per case with
`simulation.temperature_protocol="cooldown"`. It initializes velocities once at
the highest `Tstar`, runs high-temperature `ramp`, `relax`, and `sample` stages,
then repeats `ramp -> relax -> sample` while cooling down the descending
temperature schedule. Each temperature folder contains phase-specific dumps such
as `HEAV.<lcden>.ramp.*.dump`, `HEAV.<lcden>.relax.*.dump`, and
`HEAV.<lcden>.sample.*.dump`; the volume analyzer uses only `.sample.` dump
files by default. See [docs/volume_phase_experiment_manual.md](docs/volume_phase_experiment_manual.md)
for the full operating procedure and parameter rationale.

Analyze volume distributions:

```bash
python3 scripts/analyze_volume_probability.py experiments/smoke_volume_phase --method gaussian --grid-spacing 0.5 --bins 50 --tail-frames-per-tstar 200 --stride 5
```

V2 keeps the old analyzer intact and adds V6-style probability figures plus a
reusable frame cache. The first run computes per-frame volumes for every
matching `.sample.` dump file and writes `Volume_probability_frames_V2.dat`;
later `--input-mode auto` runs reuse that cache and regenerate figures/tables
without recomputing volume:

```bash
python3 scripts/analyze_volume_probability_V2.py experiments/smoke_volume_phase --method gaussian --grid-spacing 0.5 --bins 80
```

V2 outputs default to the root you pass in. If you `cd` into a simulation result
directory and run the script without a root argument, it scans that directory's
subfolders and writes the V2 `.dat` and figure files directly there.

After the full frame cache exists, changing histogram bins or frame selection is
fast and does not reread dump files:

```bash
python3 scripts/analyze_volume_probability_V2.py experiments/smoke_volume_phase --input-mode dat --bins 120 --tail-frames-per-tstar 1000 --stride 5
python3 scripts/analyze_volume_probability_V2.py experiments/smoke_volume_phase --input-mode dat --bins 60 --min-timestep 200000 --max-timestep 600000
```

V2 prints progress by default while scanning and processing dump files. Use
`--progress-every 25` to print more often on long convex-hull runs, or
`--quiet` to suppress progress messages.

Changing `--method` changes the definition of volume, so it requires a new
per-frame cache. Convex hull volume needs SciPy:

```bash
python3 -m pip install --user scipy matplotlib
python3 scripts/analyze_volume_probability_V2.py experiments/smoke_volume_phase --input-mode dump --method convex_hull --bins 80
python3 scripts/analyze_volume_probability_V2.py experiments/smoke_volume_phase --input-mode dat --method convex_hull --bins 120 --tail-frames-per-tstar 1000
```

Outputs are written under `analysis/volume_probability/`:

- `Volume_probability_summary.dat`
- `Volume_probability_bins_all_cases.dat`
- `Volume_phase_diagram.dat`

Gaussian volume works with the standard library and requires `xu yu zu` dump
columns. Convex hull volume is available as `--method convex_hull` only when
SciPy is installed.
