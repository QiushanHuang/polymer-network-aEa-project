from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

from .dump import parse_lammps_dump_frames
from .volume import convex_hull_volume, gaussian_density_volume
from .volume_probability import (
    _case_label,
    _format_value,
    _lcden_tag,
    _selected_phase_dump_paths,
    _temperature_from_tstar,
    _tstar_label,
)

OUT_DIR_DEFAULT = "."
FRAME_CACHE_DAT = "Volume_probability_frames_V2.dat"
SUMMARY_DAT = "Volume_probability_summary_V2.dat"
ALL_BINS_DAT = "Volume_probability_bins_all_cases_V2.dat"
PHASE_DAT = "Volume_phase_diagram_V2.dat"
HEATMAP_CMAP = "turbo"

VALUE_SPEC = {
    "column": "volume",
    "xlabel": "Network volume $V$",
    "ylabel": "P($V$)",
    "title": "Volume probability distributions",
}


@dataclass(frozen=True)
class VolumeProbabilityV2Result:
    frame_records: list[dict[str, object]]
    summary_records: list[dict[str, object]]
    bin_records: list[dict[str, object]]
    out_dir: Path
    input_mode: str


def analyze_volume_probability_v2(
    root_dir: str | Path,
    *,
    out_dir: str | Path,
    input_mode: str = "auto",
    method: str = "gaussian",
    bins: int = 80,
    grid_spacing: float = 0.5,
    threshold: float = math.exp(-0.5),
    gaussian_cutoff_q2: float = 6.0,
    selected_types: set[int] | None = None,
    block_size: int = 10,
    tail_frames_per_tstar: int | None = None,
    stride: int = 1,
    phase: str = "sample",
    volume_min: float | None = None,
    volume_max: float | None = None,
    smooth_window: int = 5,
    min_peak_fraction: float = 0.08,
    max_acf_lag: int = 1000,
    write_plots: bool = True,
    write_each_tstar: bool = False,
    show: bool = False,
) -> VolumeProbabilityV2Result:
    if selected_types is None:
        selected_types = {1, 2}
    _validate_common_args(
        bins=bins,
        block_size=block_size,
        stride=stride,
        tail_frames_per_tstar=tail_frames_per_tstar,
        smooth_window=smooth_window,
        min_peak_fraction=min_peak_fraction,
        max_acf_lag=max_acf_lag,
    )

    root = Path(root_dir).expanduser().resolve()
    output = Path(out_dir).expanduser()
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    resolved_mode = resolve_input_mode(input_mode, root, out_dir=output, phase=phase)
    if resolved_mode == "dat":
        cache_path = find_frame_cache(root, out_dir=output)
        if cache_path is None:
            raise ValueError(f"No {FRAME_CACHE_DAT} cache found under {output} or {root}")
        frame_records = load_frame_cache(
            cache_path,
            tail_frames_per_tstar=tail_frames_per_tstar,
            stride=stride,
            phase=phase,
        )
    elif resolved_mode == "dump":
        frame_records = build_frame_records_from_dump_tree(
            root,
            method=method,
            grid_spacing=grid_spacing,
            threshold=threshold,
            gaussian_cutoff_q2=gaussian_cutoff_q2,
            selected_types=selected_types,
            tail_frames_per_tstar=tail_frames_per_tstar,
            stride=stride,
            phase=phase,
        )
        write_frame_cache(
            output / FRAME_CACHE_DAT,
            frame_records,
            comments=[
                "Volume probability V2 frame cache.",
                "This file stores expensive per-frame volume calculations for fast plot regeneration.",
                f"method = {method}",
                f"grid_spacing = {_format_value(grid_spacing)}",
                f"threshold = {_format_value(threshold)}",
                f"gaussian_cutoff_q2 = {_format_value(gaussian_cutoff_q2)}",
                f"selected_types = {','.join(str(item) for item in sorted(selected_types))}",
                f"phase = {phase}",
                f"tail_frames_per_tstar = {tail_frames_per_tstar if tail_frames_per_tstar is not None else 'all'}",
                f"stride = {stride}",
            ],
        )
    else:
        raise ValueError("input_mode must be 'auto', 'dat', or 'dump'")

    if not frame_records:
        raise ValueError(f"No usable {phase} volume frames found under {root}")

    cases = build_case_distributions(
        frame_records,
        bins=bins,
        volume_range=(volume_min, volume_max),
        smooth_window=smooth_window,
        min_peak_fraction=min_peak_fraction,
        max_acf_lag=max_acf_lag,
        block_size=block_size,
    )
    if not cases:
        raise ValueError("No volume distributions could be built from the selected frames")

    summary_records: list[dict[str, object]] = []
    bin_records: list[dict[str, object]] = []
    per_case_bin_records: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        case_summary = summary_records_for_case(case)
        case_bins = bin_records_for_case(case)
        summary_records.extend(case_summary)
        bin_records.extend(case_bins)
        per_case_bin_records[str(case["label"])] = case_bins

    write_outputs(output, summary_records, bin_records, per_case_bin_records)
    write_phase_diagram(output / PHASE_DAT, summary_records)

    if write_plots:
        plot_cases(cases, output, write_each_tstar=write_each_tstar, show=show)

    return VolumeProbabilityV2Result(
        frame_records=frame_records,
        summary_records=summary_records,
        bin_records=bin_records,
        out_dir=output,
        input_mode=resolved_mode,
    )


def _validate_common_args(
    *,
    bins: int,
    block_size: int,
    stride: int,
    tail_frames_per_tstar: int | None,
    smooth_window: int,
    min_peak_fraction: float,
    max_acf_lag: int,
) -> None:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if tail_frames_per_tstar is not None and tail_frames_per_tstar <= 0:
        raise ValueError("tail_frames_per_tstar must be positive")
    if smooth_window <= 0:
        raise ValueError("smooth_window must be positive")
    if min_peak_fraction <= 0.0:
        raise ValueError("min_peak_fraction must be positive")
    if max_acf_lag <= 0:
        raise ValueError("max_acf_lag must be positive")


def resolve_input_mode(
    requested_mode: str,
    root_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    phase: str = "sample",
) -> str:
    if requested_mode not in {"auto", "dat", "dump"}:
        raise ValueError("input_mode must be 'auto', 'dat', or 'dump'")
    if requested_mode != "auto":
        return requested_mode

    root = Path(root_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve() if out_dir is not None else None
    if find_frame_cache(root, out_dir=output) is not None:
        return "dat"
    if any(root.rglob(f"*.{phase}.*.dump")):
        return "dump"
    return "dat"


def find_frame_cache(root_dir: str | Path, *, out_dir: str | Path | None = None) -> Path | None:
    candidates = []
    if out_dir is not None:
        output_cache = Path(out_dir).expanduser()
        if not output_cache.is_absolute():
            output_cache = Path(root_dir).expanduser().resolve() / output_cache
        output_cache = output_cache / FRAME_CACHE_DAT
        if output_cache.is_file():
            candidates.append(output_cache.resolve())

    root = Path(root_dir).expanduser().resolve()
    candidates.extend(path.resolve() for path in root.rglob(FRAME_CACHE_DAT) if path.is_file())
    if not candidates:
        return None
    return sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))[0]


def build_frame_records_from_dump_tree(
    root_dir: str | Path,
    *,
    method: str,
    grid_spacing: float,
    threshold: float,
    gaussian_cutoff_q2: float,
    selected_types: set[int],
    tail_frames_per_tstar: int | None,
    stride: int,
    phase: str,
) -> list[dict[str, object]]:
    root = Path(root_dir).expanduser().resolve()
    dump_paths = _selected_phase_dump_paths(
        sorted(root.rglob(f"*.{phase}.*.dump")),
        tail_frames_per_tstar=tail_frames_per_tstar,
        stride=stride,
    )
    records: list[dict[str, object]] = []
    for read_order, dump_path in enumerate(dump_paths):
        case_label = _case_label(dump_path)
        tstar = _tstar_label(dump_path)
        for frame in parse_lammps_dump_frames(dump_path, require_unwrapped=True):
            atoms = frame.selected_atoms(selected_types)
            volume = frame_volume(
                atoms,
                method=method,
                grid_spacing=grid_spacing,
                threshold=threshold,
                gaussian_cutoff_q2=gaussian_cutoff_q2,
                selected_types=selected_types,
            )
            records.append(
                {
                    "case_label": case_label,
                    "tstar": tstar,
                    "temperature": _temperature_from_tstar(tstar),
                    "phase": phase,
                    "timestep": frame.timestep,
                    "volume": volume,
                    "source_dump": str(dump_path),
                    "_read_order": read_order,
                }
            )
    if not records:
        raise ValueError(f"No {phase} dump frames found under {root}")
    return sorted(records, key=frame_sort_key)


def frame_volume(
    atoms: Iterable[object],
    *,
    method: str,
    grid_spacing: float,
    threshold: float,
    gaussian_cutoff_q2: float,
    selected_types: set[int],
) -> float:
    atom_tuple = tuple(atoms)
    if method == "gaussian":
        return gaussian_density_volume(
            atom_tuple,
            grid_spacing=grid_spacing,
            threshold=threshold,
            gaussian_cutoff_q2=gaussian_cutoff_q2,
            selected_types=selected_types,
        )
    if method == "convex_hull":
        return convex_hull_volume(tuple((atom.xu, atom.yu, atom.zu) for atom in atom_tuple))
    raise ValueError("method must be 'gaussian' or 'convex_hull'")


def write_frame_cache(path: str | Path, records: list[dict[str, object]], *, comments: list[str] | None = None) -> None:
    columns = [
        "row_kind",
        "case_label",
        "tstar",
        "temperature",
        "phase",
        "timestep",
        "volume",
        "source_dump",
    ]
    rows = []
    for record in records:
        row = dict(record)
        row["row_kind"] = "FRAME"
        rows.append(row)
    write_table(Path(path), columns, rows, comments=comments)


def load_frame_cache(
    path: str | Path,
    *,
    tail_frames_per_tstar: int | None = None,
    stride: int = 1,
    phase: str | None = None,
) -> list[dict[str, object]]:
    metadata, header, rows = parse_dat_file(path)
    if "row_kind" not in header:
        raise ValueError(f"{path} is not a V2 frame cache")

    records = []
    for index, row in enumerate(rows):
        if row.get("row_kind") != "FRAME":
            continue
        if phase is not None and row.get("phase", metadata.get("phase")) != phase:
            continue
        records.append(
            {
                "case_label": row["case_label"],
                "tstar": row["tstar"],
                "temperature": to_float(row.get("temperature")),
                "phase": row.get("phase", metadata.get("phase", "sample")),
                "timestep": to_int(row.get("timestep"), default=index),
                "volume": to_float(row.get("volume")),
                "source_dump": row.get("source_dump", str(path)),
                "_read_order": index,
            }
        )

    return select_tail_frame_records(records, tail_frames_per_tstar=tail_frames_per_tstar, stride=stride)


def parse_dat_file(path: str | Path) -> tuple[dict[str, str], list[str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    data_lines: list[str] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                stripped = line[1:].strip()
                if " = " in stripped:
                    key, value = stripped.split(" = ", 1)
                    metadata[key.strip()] = value.strip()
                else:
                    parts = stripped.split()
                    if parts and parts[0] in {"row_kind", "case_label"}:
                        header = parts
                continue
            data_lines.append(line.strip())

    if header is None:
        raise ValueError(f"{path} has no recognizable table header")

    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        if len(parts) > len(header):
            head = parts[: len(header) - 1]
            tail = " ".join(parts[len(header) - 1 :])
            parts = head + [tail]
        rows.append({header[index]: parts[index] for index in range(len(header))})
    return metadata, header, rows


def select_tail_frame_records(
    frame_records: list[dict[str, object]],
    *,
    tail_frames_per_tstar: int | None,
    stride: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in frame_records:
        grouped[(str(record["case_label"]), str(record["tstar"]))].append(record)

    selected = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=frame_sort_key)
        if tail_frames_per_tstar is not None:
            rows = rows[-tail_frames_per_tstar:]
        selected.extend(rows[::stride])
    return selected


def build_case_distributions(
    frame_records: list[dict[str, object]],
    *,
    bins: int,
    volume_range: tuple[float | None, float | None],
    smooth_window: int,
    min_peak_fraction: float,
    max_acf_lag: int,
    block_size: int,
) -> list[dict[str, object]]:
    records_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in frame_records:
        if is_finite(record.get("volume")):
            records_by_case[str(record["case_label"])].append(record)

    cases = []
    for case_label in sorted(records_by_case, key=tokenize_label_for_sort):
        case_records = sorted(records_by_case[case_label], key=frame_sort_key)
        values = [float(record["volume"]) for record in case_records]
        edges = determine_edges(values, bins=bins, lower=volume_range[0], upper=volume_range[1])
        if edges is None:
            continue
        centers = [0.5 * (edges[index] + edges[index + 1]) for index in range(len(edges) - 1)]

        rows_by_tstar: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in case_records:
            rows_by_tstar[str(record["tstar"])].append(record)

        by_tstar = {}
        for read_order, tstar in enumerate(sorted(rows_by_tstar, key=tstar_sort_key)):
            rows = sorted(rows_by_tstar[tstar], key=frame_sort_key)
            row_values = [float(row["volume"]) for row in rows if is_finite(row.get("volume"))]
            if not row_values:
                continue
            counts = histogram_counts(row_values, edges)
            widths = [edges[index + 1] - edges[index] for index in range(len(edges) - 1)]
            probabilities = [count / float(len(row_values)) for count in counts]
            densities = [
                probability / width if width > 0.0 else 0.0
                for probability, width in zip(probabilities, widths)
            ]
            smoothed = moving_average(densities, smooth_window)
            cumulative = cumulative_sum(probabilities)
            stats = compute_distribution_stats(
                row_values,
                centers,
                densities,
                smooth_window=smooth_window,
                min_peak_fraction=min_peak_fraction,
                max_acf_lag=max_acf_lag,
                block_size=block_size,
            )
            by_tstar[tstar] = {
                "tstar": tstar,
                "temperature": finite_or_nan(rows[0].get("temperature")),
                "sequence_index": read_order,
                "dump_min_index": min_dump_index(rows),
                "dump_max_index": max_dump_index(rows),
                "_read_order": read_order,
                "values": row_values,
                "centers": centers,
                "edges": edges,
                "counts": counts,
                "probability": probabilities,
                "density": densities,
                "smoothed_density": smoothed,
                "cumulative_probability": cumulative,
                "stats": stats,
                "frame_rows": rows,
            }
        if by_tstar:
            cases.append(
                {
                    "label": case_label,
                    "source_dat": source_label(case_records),
                    "edges": edges,
                    "centers": centers,
                    "by_tstar": dict(
                        sorted(by_tstar.items(), key=lambda item: distribution_sequence_sort_key(item[0], item[1]))
                    ),
                }
            )
    return cases


def compute_distribution_stats(
    values: list[float],
    centers: list[float],
    density: list[float],
    *,
    smooth_window: int,
    min_peak_fraction: float,
    max_acf_lag: int,
    block_size: int,
) -> dict[str, object]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError("No finite values available for distribution statistics")

    sample_count = len(finite_values)
    sample_mean = mean(finite_values)
    sample_std = stdev(finite_values) if sample_count > 1 else 0.0
    sem_naive = sample_std / math.sqrt(sample_count) if sample_count > 0 else float("nan")
    tau, n_eff = estimate_autocorrelation_time(finite_values, max_lag=max_acf_lag)
    sem_corr = sample_std / math.sqrt(n_eff) if n_eff > 0.0 else float("nan")
    q05 = percentile(finite_values, 5.0)
    q25 = percentile(finite_values, 25.0)
    q75 = percentile(finite_values, 75.0)
    q95 = percentile(finite_values, 95.0)
    centered = [value - sample_mean for value in finite_values]
    if sample_std > 0.0:
        skewness = mean([(value / sample_std) ** 3 for value in centered])
        excess_kurtosis = mean([(value / sample_std) ** 4 for value in centered]) - 3.0
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    blocks = block_means(finite_values, block_size=block_size)
    block_std = stdev(blocks) if len(blocks) > 1 else 0.0
    block_sem = block_std / math.sqrt(len(blocks)) if blocks else float("nan")
    ci_half = 1.96 * block_sem if math.isfinite(block_sem) else float("nan")

    return {
        "sample_count": sample_count,
        "block_count": len(blocks),
        "mean": sample_mean,
        "std": sample_std,
        "sem_naive": sem_naive,
        "sem_corr": sem_corr,
        "autocorr_time": tau,
        "n_eff": n_eff,
        "median": median(finite_values),
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "iqr": q75 - q25,
        "min": min(finite_values),
        "max": max(finite_values),
        "block_mean_ci_low": mean(blocks) - ci_half if blocks else float("nan"),
        "block_mean_ci_high": mean(blocks) + ci_half if blocks else float("nan"),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        **detect_peaks(centers, density, smooth_window=smooth_window, min_peak_fraction=min_peak_fraction),
    }


def summary_records_for_case(case: dict[str, object]) -> list[dict[str, object]]:
    records = []
    by_tstar = case["by_tstar"]
    assert isinstance(by_tstar, dict)
    for tstar, item in by_tstar.items():
        stats = item["stats"]
        record = {
            "case_label": case["label"],
            "source_dat": case["source_dat"],
            "tstar": tstar,
            "temperature": item["temperature"],
            "sequence_index": item.get("sequence_index"),
            "dump_min_index": item.get("dump_min_index"),
            "dump_max_index": item.get("dump_max_index"),
            "value_kind": "absolute",
            "value_column": "volume",
            "frame_selection": "selected_frames",
        }
        record.update(stats)
        records.append(record)
    return records


def bin_records_for_case(case: dict[str, object]) -> list[dict[str, object]]:
    records = []
    by_tstar = case["by_tstar"]
    assert isinstance(by_tstar, dict)
    for tstar, item in by_tstar.items():
        stats = item["stats"]
        edges = item["edges"]
        centers = item["centers"]
        max_density = max([value for value in item["density"] if math.isfinite(value)], default=float("nan"))
        for index, center in enumerate(centers):
            density = item["density"][index]
            free_energy_over_tstar = float("nan")
            free_energy_tstar = float("nan")
            if math.isfinite(density) and density > 0.0 and math.isfinite(max_density) and max_density > 0.0:
                free_energy_over_tstar = -math.log(density / max_density)
                temperature = finite_or_nan(item["temperature"])
                if math.isfinite(temperature):
                    free_energy_tstar = temperature * free_energy_over_tstar
            records.append(
                {
                    "case_label": case["label"],
                    "source_dat": case["source_dat"],
                    "tstar": tstar,
                    "temperature": item["temperature"],
                    "sequence_index": item.get("sequence_index"),
                    "dump_min_index": item.get("dump_min_index"),
                    "dump_max_index": item.get("dump_max_index"),
                    "value_kind": "absolute",
                    "value_column": "volume",
                    "bin_index": index,
                    "bin_left": edges[index],
                    "bin_center": center,
                    "bin_right": edges[index + 1],
                    "count": item["counts"][index],
                    "probability": item["probability"][index],
                    "probability_density": density,
                    "smoothed_probability_density": item["smoothed_density"][index],
                    "cumulative_probability": item["cumulative_probability"][index],
                    "effective_free_energy_over_tstar": free_energy_over_tstar,
                    "effective_free_energy_tstar": free_energy_tstar,
                    "sample_count": stats["sample_count"],
                    "frame_selection": "selected_frames",
                }
            )
    return records


def write_outputs(
    out_dir: Path,
    all_summary_records: list[dict[str, object]],
    all_bin_records: list[dict[str, object]],
    per_case_bin_records: dict[str, list[dict[str, object]]],
) -> None:
    summary_columns = [
        "case_label",
        "source_dat",
        "tstar",
        "temperature",
        "sequence_index",
        "dump_min_index",
        "dump_max_index",
        "value_kind",
        "value_column",
        "frame_selection",
        "sample_count",
        "block_count",
        "mean",
        "std",
        "sem_naive",
        "sem_corr",
        "autocorr_time",
        "n_eff",
        "median",
        "q05",
        "q25",
        "q75",
        "q95",
        "iqr",
        "min",
        "max",
        "block_mean_ci_low",
        "block_mean_ci_high",
        "skewness",
        "excess_kurtosis",
        "peak_count",
        "primary_peak_value",
        "primary_peak_density",
        "secondary_peak_value",
        "secondary_peak_density",
        "peak_separation",
        "valley_density_between_top2",
        "valley_to_lower_peak_ratio",
    ]
    bin_columns = [
        "case_label",
        "source_dat",
        "tstar",
        "temperature",
        "sequence_index",
        "dump_min_index",
        "dump_max_index",
        "value_kind",
        "value_column",
        "bin_index",
        "bin_left",
        "bin_center",
        "bin_right",
        "count",
        "probability",
        "probability_density",
        "smoothed_probability_density",
        "cumulative_probability",
        "effective_free_energy_over_tstar",
        "effective_free_energy_tstar",
        "sample_count",
        "frame_selection",
    ]
    write_table(
        out_dir / SUMMARY_DAT,
        summary_columns,
        all_summary_records,
        comments=[
            "Volume probability distribution V2 summary.",
            "Rows are derived from Volume_probability_frames_V2.dat when input-mode is dat.",
            "P(V) uses histogram probability density; peak_count is a simple smoothed-histogram diagnostic.",
        ],
    )
    write_table(
        out_dir / ALL_BINS_DAT,
        bin_columns,
        all_bin_records,
        comments=[
            "Volume probability distribution V2 bin data for all cases.",
            "probability = count / sample_count; probability_density = probability / bin_width.",
            "effective_free_energy_over_tstar = -ln(P/Pmax); effective_free_energy_tstar = -Tstar*ln(P/Pmax).",
        ],
    )
    for case_label, records in per_case_bin_records.items():
        write_table(
            out_dir / f"Volume_probability_bins_V2_{sanitize_filename(case_label)}.dat",
            bin_columns,
            records,
            comments=[f"Volume probability V2 bin data for case: {case_label}"],
        )


def write_phase_diagram(path: Path, summary_records: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in summary_records:
        grouped[(_lcden_tag(str(record["case_label"])), str(record["tstar"]))].append(record)

    rows = []
    for (lcden_tag, tstar), items in sorted(grouped.items()):
        means = [float(item["mean"]) for item in items]
        sem = stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
        rows.append(
            {
                "lcden_tag": lcden_tag,
                "tstar": tstar,
                "temperature": items[0]["temperature"],
                "seed_count": len(items),
                "mean_volume": mean(means),
                "sem_across_seeds": sem,
                "sample_count": sum(int(item["sample_count"]) for item in items),
            }
        )
    write_table(
        path,
        ["lcden_tag", "tstar", "temperature", "seed_count", "mean_volume", "sem_across_seeds", "sample_count"],
        rows,
        comments=["Volume phase diagram aggregated across seeds."],
    )


def write_table(path: Path, columns: list[str], records: list[dict[str, object]], comments: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if comments:
            for line in comments:
                handle.write(f"# {line}\n")
        handle.write("# " + " ".join(columns) + "\n")
        for record in records:
            handle.write(" ".join(serialize_value(record.get(column, float("nan"))) for column in columns) + "\n")


def plot_cases(cases: list[dict[str, object]], out_dir: Path, *, write_each_tstar: bool, show: bool) -> None:
    require_plotting()
    for case in cases:
        case_label = sanitize_filename(str(case["label"]))
        combined_path = out_dir / f"figure_all_volume_probability_views_{case_label}.png"
        plot_combined_case(case, combined_path, show=show)

        distribution_path = out_dir / f"figure1_volume_probability_distribution_{case_label}.png"
        heatmap_path = out_dir / f"figure2_volume_probability_heatmap_{case_label}.png"
        stacked_path = out_dir / f"figure3_volume_probability_stacked_{case_label}.png"
        summary_path = out_dir / f"figure4_volume_probability_summary_{case_label}.png"
        plot_single_distribution(case, distribution_path)
        plot_heatmap_figure(case, heatmap_path)
        plot_stacked_distribution_figure(case, stacked_path)
        plot_summary_figure(case, summary_path)
        if write_each_tstar:
            plot_single_tstar_diagnostics(case, out_dir)


def require_plotting() -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Volume probability V2 plotting requires matplotlib. "
            f"Install it with: python3 -m pip install matplotlib"
        ) from exc


def plot_combined_case(case: dict[str, object], out_path: Path, *, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    by_tstar = case["by_tstar"]
    assert isinstance(by_tstar, dict)
    n_tstar = len(by_tstar)
    fig_height = max(12.0, 4.5 + 0.42 * max(n_tstar, 1))
    fig = plt.figure(figsize=(18, fig_height))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.36, wspace=0.28)

    stacked_axes = plot_stacked_volume_distributions(fig, grid[:, 0], case)
    if stacked_axes:
        fig.text(0.035, 0.5, VALUE_SPEC["ylabel"], rotation=90, va="center", fontsize=10)

    ax_heatmap = fig.add_subplot(grid[0, 1])
    plot_heatmap(ax_heatmap, case, VALUE_SPEC["xlabel"], "$T^*$ - volume probability heatmap")
    ax_width_peak = fig.add_subplot(grid[1, 1])
    plot_width_peak_diagnostics(ax_width_peak, case)
    ax_summary = fig.add_subplot(grid[2, 1])
    plot_volume_summary_vs_tstar(ax_summary, case)

    fig.suptitle(f"Volume probability distribution diagnostics | {case['label']}", fontsize=13)
    fig.subplots_adjust(top=0.93, bottom=0.06)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_single_distribution(case: dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_distribution_curves(ax, case, VALUE_SPEC["xlabel"], VALUE_SPEC["ylabel"], f"{VALUE_SPEC['title']} | {case['label']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap_figure(case: dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_heatmap(ax, case, VALUE_SPEC["xlabel"], f"$T^*$ - volume probability heatmap | {case['label']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stacked_distribution_figure(case: dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_tstar = case["by_tstar"]
    assert isinstance(by_tstar, dict)
    fig_height = max(8.0, 2.5 + 0.42 * max(len(by_tstar), 1))
    fig = plt.figure(figsize=(10, fig_height))
    grid = fig.add_gridspec(1, 1)
    axes = plot_stacked_volume_distributions(fig, grid[0, 0], case)
    if axes:
        fig.text(0.02, 0.5, VALUE_SPEC["ylabel"], rotation=90, va="center", fontsize=10)
    fig.suptitle(f"Stacked volume distributions | {case['label']}", fontsize=12)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_summary_figure(case: dict[str, object], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_volume_summary_vs_tstar(ax, case)
    fig.suptitle(f"Volume summary by $T^*$ | {case['label']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_distribution_curves(ax, case: dict[str, object], xlabel: str, ylabel: str, title: str) -> None:
    from matplotlib import cm
    import matplotlib.pyplot as plt

    items = list(case["by_tstar"].values())
    cmap = plt.get_cmap("viridis")
    norm = temperature_norm(case)
    for index, item in enumerate(items):
        color = color_for_item(item, index, len(items), norm, cmap)
        ax.plot(item["centers"], item["smoothed_density"], color=color, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()
    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.set_label("$T^*$")


def plot_heatmap(ax, case: dict[str, object], xlabel: str, title: str) -> None:
    items = list(case["by_tstar"].values())
    if not items:
        ax.set_axis_off()
        return
    y_values = list(range(len(items)))
    y_edges = make_axis_edges(y_values)
    matrix = [item["smoothed_density"] for item in items]
    edges = items[0]["edges"]
    mesh = ax.pcolormesh(edges, y_edges, matrix, shading="auto", cmap=HEATMAP_CMAP)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$T^*$ sequence")
    tick_indices = list(range(len(items))) if len(items) <= 10 else unique_round_linspace(0, len(items) - 1, 8)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(
        [
            f"{items[index]['temperature']:.2f}" if is_finite(items[index]["temperature"]) else items[index]["tstar"]
            for index in tick_indices
        ]
    )
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_title(title, fontsize=11)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label("P($V$)")


def plot_stacked_volume_distributions(fig, subplot_spec, case: dict[str, object]):
    from matplotlib.ticker import AutoMinorLocator
    import matplotlib.pyplot as plt

    items = list(case["by_tstar"].values())
    if not items:
        ax = fig.add_subplot(subplot_spec)
        ax.set_axis_off()
        return []

    subgrid = subplot_spec.subgridspec(len(items), 1, hspace=0.08)
    cmap = plt.get_cmap("viridis")
    norm = temperature_norm(case)
    axes = []
    shared_ax = None
    for index, item in enumerate(items):
        ax = fig.add_subplot(subgrid[index, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        axes.append(ax)

        color = color_for_item(item, index, len(items), norm, cmap)
        ax.fill_between(item["centers"], item["smoothed_density"], color=color, alpha=0.18, linewidth=0)
        ax.plot(item["centers"], item["smoothed_density"], color=color, linewidth=1.2)
        ax.axvline(item["stats"]["mean"], color="0.15", linewidth=1.35, linestyle=":", alpha=0.85)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis="x", which="major", alpha=0.16, linewidth=0.55)
        ax.grid(True, axis="x", which="minor", alpha=0.08, linewidth=0.35)
        ax.text(
            0.01,
            0.78,
            item["tstar"],
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
        )
        if index < len(items) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(VALUE_SPEC["xlabel"])

    axes[0].set_title("P($V$) stacked by $T^*$ (dotted vertical line = mean)", fontsize=11)
    return axes


def plot_volume_summary_vs_tstar(ax, case: dict[str, object]) -> None:
    items = list(case["by_tstar"].values())
    temps = [finite_or_nan(item["temperature"]) for item in items]
    means = [float(item["stats"]["mean"]) for item in items]
    medians = [float(item["stats"]["median"]) for item in items]
    q05 = [float(item["stats"]["q05"]) for item in items]
    q25 = [float(item["stats"]["q25"]) for item in items]
    q75 = [float(item["stats"]["q75"]) for item in items]
    q95 = [float(item["stats"]["q95"]) for item in items]

    if all(math.isfinite(value) for value in temps):
        x_values = temps
        xlabel = "$T^*$"
        use_temperature_axis = True
    else:
        x_values = list(range(len(items)))
        xlabel = "Tstar index"
        use_temperature_axis = False

    ax.fill_between(x_values, q05, q95, color="tab:blue", alpha=0.12, label="5-95%")
    ax.fill_between(x_values, q25, q75, color="tab:blue", alpha=0.24, label="25-75%")
    ax.plot(x_values, medians, marker="s", linewidth=1.5, color="tab:orange", label="median")
    ax.plot(x_values, means, marker="o", linewidth=1.5, color="tab:blue", label="mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Network volume $V$")
    ax.set_title("Volume range by $T^*$", fontsize=11)
    configure_dense_summary_grid(ax, x_values, means, q05, q95, use_temperature_axis)
    apply_temperature_sequence_direction(ax, x_values, use_temperature_axis)
    ax.legend(loc="best", fontsize=8)


def plot_width_peak_diagnostics(ax, case: dict[str, object]) -> None:
    from matplotlib.ticker import AutoMinorLocator

    items = list(case["by_tstar"].values())
    temps = [finite_or_nan(item["temperature"]) for item in items]
    means = [float(item["stats"]["mean"]) for item in items]
    stds = [float(item["stats"]["std"]) for item in items]
    peaks = [float(item["stats"]["peak_count"]) for item in items]
    if all(math.isfinite(value) for value in temps):
        x_values = temps
        xlabel = "$T^*$"
        use_temperature_axis = True
    else:
        x_values = list(range(len(items)))
        xlabel = "Tstar index"
        use_temperature_axis = False

    ax.errorbar(
        x_values,
        means,
        yerr=stds,
        marker="o",
        linewidth=1.5,
        capsize=3,
        color="tab:blue",
        label="mean +/- std",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Network volume $V$")
    ax.set_title("Width and peak-count diagnostics", fontsize=11)

    ax2 = ax.twinx()
    ax2.plot(x_values, peaks, marker="s", linestyle="--", color="tab:red", linewidth=1.3, label="peak count")
    ax2.set_ylabel("Estimated peak count")
    ax2.set_ylim(bottom=0)

    configure_diagnostic_x_grid(ax, x_values, use_temperature_axis)
    apply_temperature_sequence_direction(ax, x_values, use_temperature_axis)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis="both", alpha=0.30, linewidth=0.65)
    ax.grid(True, which="minor", axis="both", alpha=0.14, linewidth=0.4)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)


def plot_single_tstar_diagnostics(case: dict[str, object], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for tstar, item in case["by_tstar"].items():
        rows = item["frame_rows"]
        x_values = [row["timestep"] for row in rows]
        y_values = [row["volume"] for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        axes[0].plot(x_values, y_values, linewidth=1.0, color="tab:blue")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Network volume $V$")
        axes[0].set_title(f"{tstar}: selected-frame volume trace", fontsize=11)
        axes[0].grid(True, alpha=0.25)

        widths = [item["edges"][index + 1] - item["edges"][index] for index in range(len(item["centers"]))]
        axes[1].bar(
            item["centers"],
            item["density"],
            width=widths,
            align="center",
            alpha=0.35,
            color="tab:blue",
            edgecolor="none",
            label="histogram density",
        )
        axes[1].plot(item["centers"], item["smoothed_density"], color="tab:red", linewidth=1.8, label="smoothed density")
        axes[1].axvline(item["stats"]["mean"], color="black", linewidth=1.2, linestyle="--", label="mean")
        axes[1].axvline(item["stats"]["median"], color="gray", linewidth=1.2, linestyle=":", label="median")
        axes[1].set_xlabel(VALUE_SPEC["xlabel"])
        axes[1].set_ylabel(VALUE_SPEC["ylabel"])
        axes[1].set_title(
            f"P($V$) | N={item['stats']['sample_count']} | peaks={item['stats']['peak_count']}",
            fontsize=11,
        )
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)

        safe_tstar = sanitize_filename(tstar)
        case_label = sanitize_filename(str(case["label"]))
        fig.suptitle(f"{case['label']} | {tstar}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_tstar}_volume_probability_{case_label}.png", dpi=200)
        plt.close(fig)


def temperature_norm(case: dict[str, object]):
    from matplotlib import colors

    temps = [finite_or_nan(item["temperature"]) for item in case["by_tstar"].values()]
    finite = [value for value in temps if math.isfinite(value)]
    if not finite or max(finite) == min(finite):
        return None
    return colors.Normalize(vmin=min(finite), vmax=max(finite))


def color_for_item(item: dict[str, object], index: int, total: int, norm, cmap):
    temperature = finite_or_nan(item.get("temperature"))
    if norm is not None and math.isfinite(temperature):
        return cmap(norm(temperature))
    return cmap(index / max(total - 1, 1))


def configure_dense_summary_grid(ax, temps: list[float], means: list[float], q05: list[float], q95: list[float], use_temperature_axis: bool) -> None:
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

    finite_temps = [value for value in temps if math.isfinite(value)]
    finite_y = [value for values in (means, q05, q95) for value in values if math.isfinite(value)]
    if len(finite_temps) >= 2 and use_temperature_axis:
        diffs = positive_diffs(sorted(set(round(value, 10) for value in finite_temps)))
        if diffs:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_minor_locator(MultipleLocator(min(diffs)))
    elif len(finite_temps) >= 2:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    if len(finite_y) >= 2:
        span = max(finite_y) - min(finite_y)
        if span > 0.0:
            major_step = nice_step(span / 6.0)
            ax.yaxis.set_major_locator(MultipleLocator(major_step))
            ax.yaxis.set_minor_locator(MultipleLocator(major_step / 2.0))
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis="both", alpha=0.30, linewidth=0.65)
    ax.grid(True, which="minor", axis="both", alpha=0.14, linewidth=0.4)


def configure_diagnostic_x_grid(ax, temps: list[float], use_temperature_axis: bool) -> None:
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

    finite_temps = [value for value in temps if math.isfinite(value)]
    if len(finite_temps) >= 2 and use_temperature_axis:
        diffs = positive_diffs(sorted(set(round(value, 10) for value in finite_temps)))
        if diffs:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_minor_locator(MultipleLocator(min(diffs)))
            return
    elif len(finite_temps) >= 2:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        return
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))


def apply_temperature_sequence_direction(ax, temps: list[float], use_temperature_axis: bool) -> None:
    if not use_temperature_axis:
        return
    finite = [value for value in temps if math.isfinite(value)]
    if len(finite) >= 2 and finite[0] > finite[-1]:
        ax.invert_xaxis()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return list(values)
    if window % 2 == 0:
        window += 1
    if window > len(values):
        window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window <= 1:
        return list(values)
    pad = window // 2
    padded = [values[0]] * pad + list(values) + [values[-1]] * pad
    smoothed = []
    for index in range(len(values)):
        segment = padded[index : index + window]
        smoothed.append(sum(segment) / float(window))
    return smoothed


def detect_peaks(centers: list[float], density: list[float], *, smooth_window: int, min_peak_fraction: float) -> dict[str, object]:
    smoothed = moving_average(density, smooth_window)
    finite = [value for value in smoothed if math.isfinite(value)]
    if not centers or not finite:
        return empty_peak_info()
    max_density = max(finite)
    if max_density <= 0.0:
        return empty_peak_info()
    threshold = max_density * min_peak_fraction
    peaks = []
    for index, value in enumerate(smoothed):
        left = smoothed[index - 1] if index > 0 else -math.inf
        right = smoothed[index + 1] if index < len(smoothed) - 1 else -math.inf
        if value >= left and value > right and value >= threshold:
            peaks.append((index, float(centers[index]), float(value)))
    if not peaks:
        index = max(range(len(smoothed)), key=lambda item: smoothed[item])
        peaks = [(index, float(centers[index]), float(smoothed[index]))]

    peaks_by_height = sorted(peaks, key=lambda item: item[2], reverse=True)
    primary = peaks_by_height[0]
    secondary = peaks_by_height[1] if len(peaks_by_height) > 1 else (None, float("nan"), float("nan"))
    valley_density = float("nan")
    valley_ratio = float("nan")
    peak_separation = float("nan")
    if secondary[0] is not None:
        lo = min(primary[0], secondary[0])
        hi = max(primary[0], secondary[0])
        if hi > lo:
            valley_density = min(smoothed[lo : hi + 1])
            lower_peak = min(primary[2], secondary[2])
            if lower_peak > 0.0:
                valley_ratio = valley_density / lower_peak
        peak_separation = abs(primary[1] - secondary[1])
    return {
        "peak_count": len(peaks),
        "primary_peak_value": primary[1],
        "primary_peak_density": primary[2],
        "secondary_peak_value": secondary[1],
        "secondary_peak_density": secondary[2],
        "peak_separation": peak_separation,
        "valley_density_between_top2": valley_density,
        "valley_to_lower_peak_ratio": valley_ratio,
    }


def empty_peak_info() -> dict[str, object]:
    return {
        "peak_count": 0,
        "primary_peak_value": float("nan"),
        "primary_peak_density": float("nan"),
        "secondary_peak_value": float("nan"),
        "secondary_peak_density": float("nan"),
        "peak_separation": float("nan"),
        "valley_density_between_top2": float("nan"),
        "valley_to_lower_peak_ratio": float("nan"),
    }


def estimate_autocorrelation_time(values: list[float], *, max_lag: int) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    n = len(finite)
    if n < 3:
        return 1.0, float(n)
    sample_mean = mean(finite)
    centered = [value - sample_mean for value in finite]
    variance = sum(value * value for value in centered) / float(n)
    if variance <= 0.0:
        return 1.0, float(n)
    positive_corr = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        numerator = sum(centered[index] * centered[index + lag] for index in range(n - lag)) / float(n - lag)
        corr = numerator / variance
        if corr <= 0.0:
            break
        positive_corr.append(corr)
    tau = 1.0 + 2.0 * sum(positive_corr)
    if not math.isfinite(tau) or tau < 1.0:
        tau = 1.0
    return tau, max(1.0, float(n) / tau)


def determine_edges(values: list[float], *, bins: int, lower: float | None, upper: float | None) -> list[float] | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    lo = min(finite) if lower is None else float(lower)
    hi = max(finite) if upper is None else float(upper)
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    if hi <= lo:
        padding = max(abs(lo) * 0.01, 1.0e-6)
        lo -= padding
        hi += padding
    width = (hi - lo) / float(bins)
    return [lo + index * width for index in range(bins + 1)]


def histogram_counts(values: list[float], edges: list[float]) -> list[int]:
    counts = [0 for _ in range(len(edges) - 1)]
    lo = edges[0]
    hi = edges[-1]
    width = (hi - lo) / float(len(counts))
    for value in values:
        if value < lo or value > hi:
            continue
        index = int((value - lo) / width) if width > 0.0 else 0
        index = min(len(counts) - 1, max(0, index))
        counts[index] += 1
    return counts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def block_means(values: list[float], *, block_size: int) -> list[float]:
    if block_size <= 1:
        return list(values)
    return [
        mean(values[index : index + block_size])
        for index in range(0, len(values), block_size)
        if values[index : index + block_size]
    ]


def cumulative_sum(values: list[float]) -> list[float]:
    total = 0.0
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def make_axis_edges(values: list[float]) -> list[float]:
    if len(values) == 1:
        width = max(abs(values[0]) * 0.02, 0.01)
        return [values[0] - width, values[0] + width]
    midpoints = [0.5 * (values[index] + values[index + 1]) for index in range(len(values) - 1)]
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return [first] + midpoints + [last]


def unique_round_linspace(start: int, stop: int, count: int) -> list[int]:
    if count <= 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return sorted(set(int(round(start + step * index)) for index in range(count)))


def positive_diffs(values: list[float]) -> list[float]:
    return [values[index + 1] - values[index] for index in range(len(values) - 1) if values[index + 1] > values[index]]


def nice_step(rough_step: float) -> float:
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1.5:
        return 1.0 * magnitude
    if normalized <= 3.0:
        return 2.0 * magnitude
    if normalized <= 7.0:
        return 5.0 * magnitude
    return 10.0 * magnitude


def min_dump_index(rows: list[dict[str, object]]) -> int | None:
    values = [dump_sequence_number(row) for row in rows if dump_sequence_number(row) is not None]
    return min(values) if values else None


def max_dump_index(rows: list[dict[str, object]]) -> int | None:
    values = [dump_sequence_number(row) for row in rows if dump_sequence_number(row) is not None]
    return max(values) if values else None


def dump_sequence_number(row: dict[str, object]) -> int | None:
    source = str(row.get("source_dump", ""))
    match = re.search(r"\.([-+]?\d+)\.dump$", source)
    if match is None:
        return None
    return int(match.group(1))


def source_label(records: list[dict[str, object]]) -> str:
    sources = sorted({str(record.get("source_dump", "")) for record in records if record.get("source_dump")})
    if not sources:
        return "unknown"
    if len(sources) == 1:
        return sources[0]
    return f"{len(sources)} source dumps"


def frame_sort_key(record: dict[str, object]) -> tuple[object, ...]:
    dump_index = dump_sequence_number(record)
    timestep = to_int(record.get("timestep"), default=None)
    read_order = to_int(record.get("_read_order"), default=0)
    return (
        str(record.get("case_label", "")),
        str(record.get("tstar", "")),
        0 if dump_index is not None else 1,
        dump_index if dump_index is not None else timestep if timestep is not None else read_order,
        timestep if timestep is not None else read_order,
        read_order,
    )


def distribution_sequence_sort_key(label: str, item: dict[str, object]) -> tuple[object, ...]:
    sequence_index = to_int(item.get("sequence_index"), default=None)
    temperature = finite_or_none(item.get("temperature"))
    if sequence_index is not None:
        return (0, sequence_index, temperature if temperature is not None else math.inf, str(label))
    return (1, temperature if temperature is not None else tstar_sort_key(label)[1], str(label))


def tstar_sort_key(label: str) -> tuple[object, ...]:
    value = _temperature_from_tstar(str(label))
    if math.isfinite(value):
        return (0, value)
    return (1, str(label))


def tokenize_label_for_sort(label: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+(?:\.\d+)?)", str(label).lower())
    tokens = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", part):
            tokens.append((1, -float(part)))
        else:
            tokens.append((0, part))
    return tuple(tokens)


def serialize_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "nan"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.8f}"


def to_float(value: object | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return float("nan")
    return float(value)


def to_int(value: object | None, *, default: int | None = None) -> int | None:
    try:
        numeric = to_float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return int(round(numeric))


def finite_or_none(value: object | None) -> float | None:
    numeric = finite_or_nan(value)
    return numeric if math.isfinite(numeric) else None


def finite_or_nan(value: object | None) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def is_finite(value: object | None) -> bool:
    return math.isfinite(finite_or_nan(value))


def sanitize_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build V6-style P(V) probability plots from cooldown sample dumps or cached V2 volume dat."
    )
    parser.add_argument("root", nargs="?", default=".", help="Suite/result root. Default: current directory.")
    parser.add_argument(
        "--out-dir",
        default=OUT_DIR_DEFAULT,
        help="Output directory. Relative paths are resolved under root. Default: root directory itself.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "dat", "dump"),
        default="auto",
        help=f"auto uses {FRAME_CACHE_DAT} when present, otherwise computes from dump files.",
    )
    parser.add_argument("--method", choices=("gaussian", "convex_hull"), default="gaussian")
    parser.add_argument("--phase", default="sample", help="Cooldown dump phase to analyze. Default: sample.")
    parser.add_argument("--types", type=int, nargs="+", default=[1, 2], help="Particle types used for volume. Default: 1 2.")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins.")
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=math.exp(-0.5))
    parser.add_argument("--gaussian-cutoff-q2", type=float, default=6.0)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--tail-frames-per-tstar", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--volume-min", type=float, default=None)
    parser.add_argument("--volume-max", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-peak-fraction", type=float, default=0.08)
    parser.add_argument("--max-acf-lag", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true", help="Write dat outputs only.")
    parser.add_argument("--write-each-tstar", action="store_true", help="Also write one diagnostic PNG per Tstar.")
    parser.add_argument("--show", action="store_true", help="Open each combined matplotlib window.")
    return parser


def run_analysis(args: argparse.Namespace) -> VolumeProbabilityV2Result:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")
    return analyze_volume_probability_v2(
        root,
        out_dir=args.out_dir,
        input_mode=args.input_mode,
        method=args.method,
        bins=args.bins,
        grid_spacing=args.grid_spacing,
        threshold=args.threshold,
        gaussian_cutoff_q2=args.gaussian_cutoff_q2,
        selected_types=set(args.types),
        block_size=args.block_size,
        tail_frames_per_tstar=args.tail_frames_per_tstar,
        stride=args.stride,
        phase=args.phase,
        volume_min=args.volume_min,
        volume_max=args.volume_max,
        smooth_window=args.smooth_window,
        min_peak_fraction=args.min_peak_fraction,
        max_acf_lag=args.max_acf_lag,
        write_plots=not args.no_plots,
        write_each_tstar=args.write_each_tstar,
        show=args.show,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = run_analysis(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[INFO] input mode = {result.input_mode}")
    print(f"[INFO] out_dir = {result.out_dir}")
    print(f"[INFO] frames = {len(result.frame_records)}")
    print(f"[INFO] summary = {result.out_dir / SUMMARY_DAT}")
    print(f"[INFO] bins = {result.out_dir / ALL_BINS_DAT}")
    print(f"[INFO] cache = {result.out_dir / FRAME_CACHE_DAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
