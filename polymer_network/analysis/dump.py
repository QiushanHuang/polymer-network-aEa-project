from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DumpAtom:
    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float
    xu: float
    yu: float
    zu: float
    quat: tuple[float, float, float, float]
    shape: tuple[float, float, float]
    mass: float


@dataclass(frozen=True)
class DumpFrame:
    source_path: Path
    timestep: int
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    columns: tuple[str, ...]
    atoms: tuple[DumpAtom, ...]

    def selected_atoms(self, selected_types: set[int]) -> tuple[DumpAtom, ...]:
        return tuple(atom for atom in self.atoms if atom.atom_type in selected_types)


def parse_lammps_dump_frames(
    path: str | Path,
    *,
    require_unwrapped: bool = True,
) -> Iterable[DumpFrame]:
    dump_path = Path(path)
    lines = dump_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("ITEM: TIMESTEP"):
            index += 1
            continue
        timestep = int(lines[index + 1].strip())
        if not lines[index + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"{dump_path} missing NUMBER OF ATOMS after timestep {timestep}")
        atom_count = int(lines[index + 3].strip())
        if not lines[index + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{dump_path} missing box bounds after timestep {timestep}")
        box_bounds = (
            _parse_bound(lines[index + 5]),
            _parse_bound(lines[index + 6]),
            _parse_bound(lines[index + 7]),
        )
        atom_header = lines[index + 8]
        if not atom_header.startswith("ITEM: ATOMS"):
            raise ValueError(f"{dump_path} missing ATOMS header after timestep {timestep}")
        columns = tuple(atom_header.split()[2:])
        if require_unwrapped and not {"xu", "yu", "zu"}.issubset(columns):
            raise ValueError(f"{dump_path} timestep {timestep} must contain xu yu zu columns")
        col_index = {name: position for position, name in enumerate(columns)}
        atoms = []
        start = index + 9
        stop = start + atom_count
        for raw in lines[start:stop]:
            parts = raw.split()
            if len(parts) < len(columns):
                raise ValueError(f"{dump_path} has a short atom row at timestep {timestep}: {raw}")
            atoms.append(_parse_atom(parts, col_index))
        yield DumpFrame(
            source_path=dump_path,
            timestep=timestep,
            box_bounds=box_bounds,
            columns=columns,
            atoms=tuple(sorted(atoms, key=lambda atom: atom.atom_id)),
        )
        index = stop


def _parse_bound(line: str) -> tuple[float, float]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid box bound line: {line}")
    return (float(parts[0]), float(parts[1]))


def _parse_atom(parts: list[str], col_index: dict[str, int]) -> DumpAtom:
    def value(name: str, default: float = 0.0) -> float:
        position = col_index.get(name)
        if position is None:
            return default
        return float(parts[position])

    x = value("x")
    y = value("y")
    z = value("z")
    return DumpAtom(
        atom_id=int(value("id")),
        atom_type=int(value("type")),
        x=x,
        y=y,
        z=z,
        xu=value("xu", x),
        yu=value("yu", y),
        zu=value("zu", z),
        quat=(
            value("quatw", 1.0),
            value("quati", 0.0),
            value("quatj", 0.0),
            value("quatk", 0.0),
        ),
        shape=(
            value("shapex", 1.0),
            value("shapey", 1.0),
            value("shapez", 1.0),
        ),
        mass=value("mass", 1.0),
    )
