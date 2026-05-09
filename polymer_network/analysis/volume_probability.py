from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

from .dump import DumpAtom, parse_lammps_dump_frames
from .volume import convex_hull_volume, gaussian_density_volume


@dataclass(frozen=True)
class VolumeAnalysisResult:
    frame_records: list[dict[str, object]]
    summary_records: list[dict[str, object]]
    bin_records: list[dict[str, object]]
    out_dir: Path


def analyze_volume_probability(
    root_dir: str | Path,
    *,
    out_dir: str | Path,
    method: str = "gaussian",
    bins: int = 50,
    grid_spacing: float = 0.5,
    threshold: float = math.exp(-0.5),
    gaussian_cutoff_q2: float = 6.0,
    selected_types: set[int] | None = None,
    block_size: int = 10,
    tail_frames_per_tstar: int | None = None,
    stride: int = 1,
    phase: str = "sample",
) -> VolumeAnalysisResult:
    if selected_types is None:
        selected_types = {1, 2}
    if bins <= 0:
        raise ValueError("bins must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if tail_frames_per_tstar is not None and tail_frames_per_tstar <= 0:
        raise ValueError("tail_frames_per_tstar must be positive")
    root = Path(root_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame_records = []
    dump_paths = _selected_phase_dump_paths(
        sorted(root.rglob(f"*.{phase}.*.dump")),
        tail_frames_per_tstar=tail_frames_per_tstar,
        stride=stride,
    )
    for dump_path in dump_paths:
        case_label = _case_label(dump_path)
        tstar = _tstar_label(dump_path)
        for frame in parse_lammps_dump_frames(dump_path, require_unwrapped=True):
            atoms = frame.selected_atoms(selected_types)
            volume = _frame_volume(
                atoms,
                method=method,
                grid_spacing=grid_spacing,
                threshold=threshold,
                gaussian_cutoff_q2=gaussian_cutoff_q2,
                selected_types=selected_types,
            )
            frame_records.append(
                {
                    "case_label": case_label,
                    "tstar": tstar,
                    "temperature": _temperature_from_tstar(tstar),
                    "phase": phase,
                    "timestep": frame.timestep,
                    "volume": volume,
                    "source_dump": str(dump_path),
                }
            )
    if not frame_records:
        raise ValueError(f"No {phase} dump frames found under {root}")

    summary_records = _summary_records(frame_records, block_size=block_size)
    bin_records = _bin_records(frame_records, bins=bins)
    _write_summary(
        output / "Volume_probability_summary.dat",
        summary_records,
        method=method,
        grid_spacing=grid_spacing,
        threshold=threshold,
        gaussian_cutoff_q2=gaussian_cutoff_q2,
        selected_types=selected_types,
        block_size=block_size,
        tail_frames_per_tstar=tail_frames_per_tstar,
        stride=stride,
        phase=phase,
    )
    _write_bins(output / "Volume_probability_bins_all_cases.dat", bin_records)
    _write_phase(output / "Volume_phase_diagram.dat", summary_records)
    return VolumeAnalysisResult(
        frame_records=frame_records,
        summary_records=summary_records,
        bin_records=bin_records,
        out_dir=output,
    )


def _frame_volume(
    atoms: Iterable[DumpAtom],
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


def _summary_records(frame_records: list[dict[str, object]], *, block_size: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in frame_records:
        key = (str(record["case_label"]), str(record["tstar"]))
        grouped.setdefault(key, []).append(record)

    records = []
    for (case_label, tstar), rows in sorted(grouped.items()):
        values = [float(row["volume"]) for row in sorted(rows, key=lambda item: int(item["timestep"]))]
        blocks = _block_means(values, block_size=block_size)
        block_std = stdev(blocks) if len(blocks) > 1 else 0.0
        block_sem = block_std / math.sqrt(len(blocks)) if blocks else float("nan")
        ci_half = 1.96 * block_sem if math.isfinite(block_sem) else float("nan")
        records.append(
            {
                "case_label": case_label,
                "tstar": tstar,
                "temperature": _temperature_from_tstar(tstar),
                "sample_count": len(values),
                "block_count": len(blocks),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "block_mean_ci_low": mean(blocks) - ci_half if blocks else float("nan"),
                "block_mean_ci_high": mean(blocks) + ci_half if blocks else float("nan"),
            }
        )
    return records


def _bin_records(frame_records: list[dict[str, object]], *, bins: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in frame_records:
        key = (str(record["case_label"]), str(record["tstar"]))
        grouped.setdefault(key, []).append(float(record["volume"]))

    records = []
    for (case_label, tstar), values in sorted(grouped.items()):
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            lo -= 0.5
            hi += 0.5
        width = (hi - lo) / bins
        counts = [0 for _ in range(bins)]
        for value in values:
            index = min(bins - 1, max(0, int((value - lo) / width)))
            counts[index] += 1
        for index, count in enumerate(counts):
            left = lo + index * width
            right = left + width
            probability = count / len(values)
            density = probability / width
            records.append(
                {
                    "case_label": case_label,
                    "tstar": tstar,
                    "temperature": _temperature_from_tstar(tstar),
                    "bin_index": index,
                    "bin_left": left,
                    "bin_center": 0.5 * (left + right),
                    "bin_right": right,
                    "count": count,
                    "probability": probability,
                    "probability_density": density,
                }
            )
    return records


def _write_summary(
    path: Path,
    records: list[dict[str, object]],
    *,
    method: str,
    grid_spacing: float,
    threshold: float,
    gaussian_cutoff_q2: float,
    selected_types: set[int],
    block_size: int,
    tail_frames_per_tstar: int | None,
    stride: int,
    phase: str,
) -> None:
    fields = [
        "case_label",
        "tstar",
        "temperature",
        "sample_count",
        "block_count",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "block_mean_ci_low",
        "block_mean_ci_high",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# method = {method}\n")
        handle.write(f"# grid_spacing = {_format_value(grid_spacing)}\n")
        handle.write(f"# threshold = {_format_value(threshold)}\n")
        handle.write(f"# gaussian_cutoff_q2 = {_format_value(gaussian_cutoff_q2)}\n")
        handle.write(f"# selected_types = {','.join(str(item) for item in sorted(selected_types))}\n")
        handle.write(f"# block_size = {block_size}\n")
        handle.write(f"# frame_selection = {phase}\n")
        handle.write(f"# tail_frames_per_tstar = {tail_frames_per_tstar if tail_frames_per_tstar is not None else 'all'}\n")
        handle.write(f"# stride = {stride}\n")
        handle.write("# " + " ".join(fields) + "\n")
        for record in records:
            handle.write(" ".join(_format_value(record[field]) for field in fields) + "\n")


def _write_bins(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "case_label",
        "tstar",
        "temperature",
        "bin_index",
        "bin_left",
        "bin_center",
        "bin_right",
        "count",
        "probability",
        "probability_density",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(fields) + "\n")
        for record in records:
            handle.write(" ".join(_format_value(record[field]) for field in fields) + "\n")


def _write_phase(path: Path, records: list[dict[str, object]]) -> None:
    phase_records = _phase_records(records)
    fields = [
        "lcden_tag",
        "tstar",
        "temperature",
        "seed_count",
        "mean_volume",
        "sem_across_seeds",
        "sample_count",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# " + " ".join(fields) + "\n")
        for record in phase_records:
            handle.write(" ".join(_format_value(record[field]) for field in fields) + "\n")


def _phase_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        lcden = _lcden_tag(str(record["case_label"]))
        key = (lcden, str(record["tstar"]))
        grouped.setdefault(key, []).append(record)

    phase_records = []
    for (lcden, tstar), rows in sorted(grouped.items()):
        means = [float(row["mean"]) for row in rows]
        sem = stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
        phase_records.append(
            {
                "lcden_tag": lcden,
                "tstar": tstar,
                "temperature": rows[0]["temperature"],
                "seed_count": len(rows),
                "mean_volume": mean(means),
                "sem_across_seeds": sem,
                "sample_count": sum(int(row["sample_count"]) for row in rows),
            }
        )
    return phase_records


def _lcden_tag(case_label: str) -> str:
    match = re.search(r"(lcden\d+)", case_label)
    if match is None:
        return case_label
    return match.group(1)


def _block_means(values: list[float], *, block_size: int) -> list[float]:
    if block_size <= 1:
        return list(values)
    return [
        mean(values[index : index + block_size])
        for index in range(0, len(values), block_size)
        if values[index : index + block_size]
    ]


def _selected_phase_dump_paths(
    paths: list[Path],
    *,
    tail_frames_per_tstar: int | None,
    stride: int,
) -> list[Path]:
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        grouped.setdefault((_case_label(path), _tstar_label(path)), []).append(path)
    selected = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=_dump_sort_key)
        if tail_frames_per_tstar is not None:
            items = items[-tail_frames_per_tstar:]
        selected.extend(items[::stride])
    return selected


def _dump_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\.([-+]?\d+)\.dump$", path.name)
    if match is None:
        return (0, path.name)
    return (int(match.group(1)), path.name)


def _case_label(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("lcden"):
            return part
    if "result" in path.parts:
        index = path.parts.index("result")
        if index > 0:
            return path.parts[index - 1]
    return path.parent.parent.name


def _tstar_label(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("Tstar_"):
            return part
    return "Tstar_nan"


def _temperature_from_tstar(tstar: str) -> float:
    match = re.search(r"Tstar_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", tstar)
    if match is None:
        return float("nan")
    return float(match.group(1))


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.8g}"
