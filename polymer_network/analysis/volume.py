from __future__ import annotations

import math
from typing import Iterable

from .dump import DumpAtom


def gaussian_density_volume(
    atoms: Iterable[DumpAtom],
    *,
    grid_spacing: float = 0.5,
    threshold: float = math.exp(-0.5),
    selected_types: set[int] | None = None,
    gaussian_cutoff_q2: float = 6.0,
) -> float:
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if gaussian_cutoff_q2 <= 0.0:
        raise ValueError("gaussian_cutoff_q2 must be positive")

    selected = [
        atom
        for atom in atoms
        if selected_types is None or atom.atom_type in selected_types
    ]
    if not selected:
        raise ValueError("No atoms selected for volume calculation")

    density_by_voxel: dict[tuple[int, int, int], float] = {}
    origin_x = min(atom.xu for atom in selected)
    origin_y = min(atom.yu for atom in selected)
    origin_z = min(atom.zu for atom in selected)
    cutoff_radius_factor = math.sqrt(gaussian_cutoff_q2)
    for atom in selected:
        radius = max(atom.shape) * 0.5 * cutoff_radius_factor
        ix_min = math.floor((atom.xu - radius - origin_x) / grid_spacing)
        ix_max = math.ceil((atom.xu + radius - origin_x) / grid_spacing)
        iy_min = math.floor((atom.yu - radius - origin_y) / grid_spacing)
        iy_max = math.ceil((atom.yu + radius - origin_y) / grid_spacing)
        iz_min = math.floor((atom.zu - radius - origin_z) / grid_spacing)
        iz_max = math.ceil((atom.zu + radius - origin_z) / grid_spacing)
        for ix in range(ix_min, ix_max + 1):
            x = origin_x + ix * grid_spacing
            for iy in range(iy_min, iy_max + 1):
                y = origin_y + iy * grid_spacing
                for iz in range(iz_min, iz_max + 1):
                    z = origin_z + iz * grid_spacing
                    q2 = ellipsoid_metric_q2(atom, x, y, z)
                    if q2 <= gaussian_cutoff_q2:
                        key = (ix, iy, iz)
                        density_by_voxel[key] = density_by_voxel.get(key, 0.0) + math.exp(-0.5 * q2)
    occupied = sum(1 for density in density_by_voxel.values() if density >= threshold)
    return occupied * (grid_spacing ** 3)


def convex_hull_volume(points: Iterable[tuple[float, float, float]]) -> float:
    try:
        from scipy.spatial import ConvexHull  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SciPy is required for convex hull volume; use method='gaussian' or install scipy."
        ) from exc

    point_list = list(points)
    if len(point_list) < 4:
        raise ValueError("Convex hull volume requires at least four 3D points")
    hull = ConvexHull(point_list)
    return float(hull.volume)


def ellipsoid_metric_q2(atom: DumpAtom, x: float, y: float, z: float) -> float:
    ax = max(atom.shape[0] * 0.5, 1.0e-12)
    ay = max(atom.shape[1] * 0.5, 1.0e-12)
    az = max(atom.shape[2] * 0.5, 1.0e-12)
    local_x, local_y, local_z = _rotate_global_to_local(
        (x - atom.xu, y - atom.yu, z - atom.zu),
        atom.quat,
    )
    return (local_x / ax) ** 2 + (local_y / ay) ** 2 + (local_z / az) ** 2


def _is_inside_gaussian_union(
    x: float,
    y: float,
    z: float,
    atoms: list[DumpAtom],
    q2_limit: float,
) -> bool:
    for atom in atoms:
        if ellipsoid_metric_q2(atom, x, y, z) <= q2_limit:
            return True
    return False


def _rotate_global_to_local(
    vector: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return vector
    w, x, y, z = (w / norm, -x / norm, -y / norm, -z / norm)
    vx, vy, vz = vector
    # q * v * q^-1 for the conjugated orientation quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)
