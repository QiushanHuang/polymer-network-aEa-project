from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Literal


TopologyMode = Literal["unified_lattice", "2D_surface", "3D_cubic"]
Orientation = Literal["horizontal", "vertical", "x", "y", "z"]


@dataclass(frozen=True)
class NetworkConfig:
    topology_mode: TopologyMode = "unified_lattice"
    cells_x: int = 7
    cells_y: int = 7
    cells_z: int = 1
    horizontal_s_count: int = 7
    vertical_s_count: int = 7
    x_s_count: int | None = None
    y_s_count: int | None = None
    z_s_count: int = 7
    insertion_density: float = 0.25
    horizontal_ratio: float = 0.5
    axis_insertion_weights: dict[str, float] | None = None
    axis_insertion_ratios: dict[str, float] | None = None
    seed: int | None = 12345
    contact_gap: float = 0.1
    box_padding: float = 10.0
    z_plane: float | None = None

    s_diameter: float = 1.0
    e_long_diameter: float = 3.0
    e_short_diameter: float = 1.0
    anchor_diameter: float = 0.00001

    mass_e: float = 1.0
    mass_s: float = 1.0 / 3.0
    mass_anchor: float = 0.00001

    geometry_mode: Literal["preserve-grid"] = "preserve-grid"

    def validate(self) -> None:
        if self.topology_mode not in ("unified_lattice", "2D_surface", "3D_cubic"):
            raise ValueError("topology_mode must be 'unified_lattice', '2D_surface', or '3D_cubic'")
        if self.cells_x <= 0 or self.cells_y <= 0 or self.cells_z <= 0:
            raise ValueError("cells_x, cells_y, and cells_z must be positive")
        if self.resolved_x_s_count < 5 or self.resolved_y_s_count < 5:
            raise ValueError("x/y S counts must be at least 5")
        if self.has_z_segments and self.resolved_z_s_count < 5:
            raise ValueError("z_s_count must be at least 5 when z segments are enabled")
        if not 0.0 <= self.insertion_density <= 1.0:
            raise ValueError("insertion_density must be in [0, 1]")
        if not 0.0 <= self.horizontal_ratio <= 1.0:
            raise ValueError("horizontal_ratio must be in [0, 1]")
        axis_ratios = self._axis_ratios()
        if not self.has_z_segments and axis_ratios["z"] > 0.0:
            raise ValueError(
                "z insertion weight requires z-directed segments; "
                "set cells_z > 1 or set the z insertion weight to 0"
            )
        if self.geometry_mode != "preserve-grid":
            raise ValueError("Only geometry_mode='preserve-grid' is currently supported")

    @property
    def s_spacing(self) -> float:
        return self.s_diameter + self.contact_gap

    @property
    def e_major_radius(self) -> float:
        return self.e_long_diameter / 2.0

    @property
    def s_radius(self) -> float:
        return self.s_diameter / 2.0

    @property
    def target_se_distance(self) -> float:
        return self.e_major_radius + self.s_radius + self.contact_gap

    @property
    def resolved_x_s_count(self) -> int:
        return self.x_s_count if self.x_s_count is not None else self.horizontal_s_count

    @property
    def resolved_y_s_count(self) -> int:
        return self.y_s_count if self.y_s_count is not None else self.vertical_s_count

    @property
    def resolved_z_s_count(self) -> int:
        return self.z_s_count

    @property
    def has_z_segments(self) -> bool:
        if self.topology_mode == "2D_surface":
            return False
        if self.topology_mode == "3D_cubic":
            return True
        return self.cells_z > 1

    @property
    def z_grid_layers(self) -> int:
        if self.topology_mode == "2D_surface":
            return 1
        if self.topology_mode == "3D_cubic":
            return self.cells_z + 1
        if self.cells_z == 1:
            return 1
        return self.cells_z + 1

    @property
    def z_cell_count(self) -> int:
        if self.topology_mode == "2D_surface":
            return 0
        if self.topology_mode == "3D_cubic":
            return self.cells_z
        if self.cells_z == 1:
            return 0
        return self.cells_z

    @property
    def effective_axis_insertion_weights(self) -> dict[str, float]:
        return self._axis_weights()

    @property
    def normalized_axis_insertion_ratios(self) -> dict[str, float]:
        return self._axis_ratios()

    @property
    def effective_horizontal_ratio(self) -> float:
        ratios = self.normalized_axis_insertion_ratios
        xy_total = ratios["x"] + ratios["y"]
        if xy_total <= 0.0:
            return 0.0
        return ratios["x"] / xy_total

    def _axis_weights(self) -> dict[str, float]:
        source = self.axis_insertion_weights
        if source is None:
            source = self.axis_insertion_ratios
        if source is None:
            if self.topology_mode == "2D_surface":
                return {
                    "x": self.horizontal_ratio,
                    "y": 1.0 - self.horizontal_ratio,
                    "z": 0.0,
                }
            if not self.has_z_segments:
                return {"x": 1.0, "y": 1.0, "z": 0.0}
            return {"x": 1.0, "y": 1.0, "z": 1.0}

        weights = {"x": 0.0, "y": 0.0, "z": 0.0}
        aliases = {
            "horizontal": "x",
            "vertical": "y",
            "depth": "z",
        }
        for key, value in source.items():
            axis = aliases.get(key, key)
            if axis not in weights:
                raise ValueError(f"Unknown insertion axis: {key}")
            weight = float(value)
            if weight < 0.0:
                raise ValueError("axis insertion weights must be non-negative")
            weights[axis] += weight
        return weights

    def _axis_ratios(self) -> dict[str, float]:
        weights = self._axis_weights()
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("axis insertion weights must contain a positive weight")
        return {axis: value / total for axis, value in weights.items()}


@dataclass(frozen=True)
class Site:
    site_id: int
    ix: int
    iy: int
    iz: int
    x: float
    y: float
    z: float
    is_cross: bool


@dataclass(frozen=True)
class Segment:
    segment_id: str
    orientation: Orientation
    site_ids: tuple[int, ...]


@dataclass(frozen=True)
class Atom:
    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float
    molecule_id: int = 0
    label: str = ""
    site_id: int | None = None
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Bond:
    bond_id: int
    bond_type: int
    atom1: int
    atom2: int


@dataclass(frozen=True)
class Insertion:
    insertion_id: int
    segment_id: str
    orientation: Orientation
    start_index: int
    replaced_site_ids: frozenset[int]
    left_neighbor_site_id: int
    right_neighbor_site_id: int
    left_neighbor_atom_id: int
    right_neighbor_atom_id: int
    left_anchor_atom_id: int
    e_atom_id: int
    right_anchor_atom_id: int
    molecule_id: int
    target_se_distance: float
    actual_left_se_distance: float
    actual_right_se_distance: float


@dataclass
class PolymerNetwork:
    config: NetworkConfig
    atoms: list[Atom]
    bonds: list[Bond]
    insertions: list[Insertion]
    sites: dict[int, Site]
    segments: list[Segment]
    cross_site_ids: frozenset[int]
    site_atom_ids: dict[int, int]
    box_length: float
    metadata: dict[str, object] = field(default_factory=dict)

    def count_atoms(self, atom_type: int) -> int:
        return sum(1 for atom in self.atoms if atom.atom_type == atom_type)

    def count_insertions(self, orientation: Orientation) -> int:
        aliases = {
            "horizontal": {"horizontal", "x"},
            "x": {"horizontal", "x"},
            "vertical": {"vertical", "y"},
            "y": {"vertical", "y"},
            "z": {"z"},
        }
        names = aliases.get(orientation, {orientation})
        return sum(1 for insertion in self.insertions if insertion.orientation in names)

    def atom_by_id(self, atom_id: int) -> Atom:
        for atom in self.atoms:
            if atom.atom_id == atom_id:
                return atom
        raise KeyError(atom_id)

    def distance(self, atom1: int, atom2: int) -> float:
        a = self.atom_by_id(atom1)
        b = self.atom_by_id(atom2)
        return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


@dataclass(frozen=True)
class _InsertionPlan:
    insertion_id: int
    segment: Segment
    start_index: int
    replaced_site_ids: frozenset[int]
    left_neighbor_site_id: int
    right_neighbor_site_id: int


def build_polymer_network(config: NetworkConfig) -> PolymerNetwork:
    config.validate()

    sites, segments = _build_grid(config)
    cross_site_ids = frozenset(site.site_id for site in sites.values() if site.is_cross)
    plans = _select_insertions(config, segments, cross_site_ids)
    replaced_site_ids = frozenset(
        site_id for plan in plans for site_id in plan.replaced_site_ids
    )

    atoms: list[Atom] = []
    site_atom_ids: dict[int, int] = {}
    next_atom_id = 1

    for site in sorted(sites.values(), key=lambda item: (item.iz, item.iy, item.ix)):
        if site.site_id in replaced_site_ids:
            continue
        atoms.append(
            Atom(
                atom_id=next_atom_id,
                atom_type=2,
                x=site.x,
                y=site.y,
                z=site.z,
                label="S",
                site_id=site.site_id,
            )
        )
        site_atom_ids[site.site_id] = next_atom_id
        next_atom_id += 1

    insertion_records: list[Insertion] = []
    for plan in plans:
        axis = _axis(plan.segment.orientation)
        left_site = sites[plan.left_neighbor_site_id]
        right_site = sites[plan.right_neighbor_site_id]
        e_x = 0.5 * (left_site.x + right_site.x)
        e_y = 0.5 * (left_site.y + right_site.y)
        e_z = 0.5 * (left_site.z + right_site.z)
        half = config.e_major_radius

        left_anchor = (
            e_x - axis[0] * half,
            e_y - axis[1] * half,
            e_z - axis[2] * half,
        )
        right_anchor = (
            e_x + axis[0] * half,
            e_y + axis[1] * half,
            e_z + axis[2] * half,
        )
        molecule_id = plan.insertion_id
        left_anchor_id = next_atom_id
        atoms.append(
            Atom(
                atom_id=left_anchor_id,
                atom_type=3,
                x=left_anchor[0],
                y=left_anchor[1],
                z=left_anchor[2],
                molecule_id=molecule_id,
                label="aL",
                quaternion=_quaternion_for_orientation(plan.segment.orientation),
            )
        )
        next_atom_id += 1

        e_atom_id = next_atom_id
        atoms.append(
            Atom(
                atom_id=e_atom_id,
                atom_type=1,
                x=e_x,
                y=e_y,
                z=e_z,
                molecule_id=molecule_id,
                label="E",
                quaternion=_quaternion_for_orientation(plan.segment.orientation),
            )
        )
        next_atom_id += 1

        right_anchor_id = next_atom_id
        atoms.append(
            Atom(
                atom_id=right_anchor_id,
                atom_type=3,
                x=right_anchor[0],
                y=right_anchor[1],
                z=right_anchor[2],
                molecule_id=molecule_id,
                label="aR",
                quaternion=_quaternion_for_orientation(plan.segment.orientation),
            )
        )
        next_atom_id += 1

        left_atom_id = site_atom_ids[plan.left_neighbor_site_id]
        right_atom_id = site_atom_ids[plan.right_neighbor_site_id]
        left_distance = math.dist((left_site.x, left_site.y, left_site.z), (e_x, e_y, e_z))
        right_distance = math.dist((right_site.x, right_site.y, right_site.z), (e_x, e_y, e_z))
        insertion_records.append(
            Insertion(
                insertion_id=plan.insertion_id,
                segment_id=plan.segment.segment_id,
                orientation=plan.segment.orientation,
                start_index=plan.start_index,
                replaced_site_ids=plan.replaced_site_ids,
                left_neighbor_site_id=plan.left_neighbor_site_id,
                right_neighbor_site_id=plan.right_neighbor_site_id,
                left_neighbor_atom_id=left_atom_id,
                right_neighbor_atom_id=right_atom_id,
                left_anchor_atom_id=left_anchor_id,
                e_atom_id=e_atom_id,
                right_anchor_atom_id=right_anchor_id,
                molecule_id=molecule_id,
                target_se_distance=config.target_se_distance,
                actual_left_se_distance=left_distance,
                actual_right_se_distance=right_distance,
            )
        )

    bonds = _build_bonds(segments, replaced_site_ids, site_atom_ids, insertion_records)
    box_length = _compute_box_length(atoms, config.box_padding)
    metadata = {
        "generator": "polymer_network",
        "topology_mode": config.topology_mode,
        "geometry_mode": config.geometry_mode,
        "contact_gap": config.contact_gap,
        "s_s_initial_distance": config.s_spacing,
        "target_s_e_distance": config.target_se_distance,
        "insertion_density_basis": "total eligible non-junction network segments",
        "insertion_count_rounding": "half-up",
        "seed_mode": "random" if config.seed is None else "fixed",
        "axis_insertion_weight_semantics": (
            "raw x/y/z weights are normalized internally to split the global "
            "insertion_density target"
        ),
        "axis_insertion_weights": config.effective_axis_insertion_weights,
        "axis_insertion_ratios": config.normalized_axis_insertion_ratios,
        "segment_counts_by_axis": _count_segments_by_axis(segments),
        "junction_site_count": len(cross_site_ids),
        "note": (
            "preserve-grid mode keeps every network junction node fixed; "
            "therefore aEa fragments that replace three S beads can have an initial S-E "
            "distance different from the target contact distance."
        ),
    }
    return PolymerNetwork(
        config=config,
        atoms=atoms,
        bonds=bonds,
        insertions=insertion_records,
        sites=sites,
        segments=segments,
        cross_site_ids=cross_site_ids,
        site_atom_ids=site_atom_ids,
        box_length=box_length,
        metadata=metadata,
    )


def _build_grid(config: NetworkConfig) -> tuple[dict[int, Site], list[Segment]]:
    if config.topology_mode == "2D_surface":
        return _build_surface_grid(config)
    if config.topology_mode == "3D_cubic":
        return _build_cubic_grid(
            config,
            z_grid_layers=config.cells_z + 1,
            z_cell_count=config.cells_z,
        )
    return _build_cubic_grid(
        config,
        z_grid_layers=config.z_grid_layers,
        z_cell_count=config.z_cell_count,
    )


def _build_surface_grid(config: NetworkConfig) -> tuple[dict[int, Site], list[Segment]]:
    hx = config.resolved_x_s_count - 1
    vy = config.resolved_y_s_count - 1
    max_ix = config.cells_x * hx
    max_iy = config.cells_y * vy
    z_plane = config.z_plane
    if z_plane is None:
        z_plane = config.box_padding

    keys: set[tuple[int, int]] = set()
    for row in range(config.cells_y + 1):
        iy = row * vy
        for ix in range(max_ix + 1):
            keys.add((ix, iy))
    for col in range(config.cells_x + 1):
        ix = col * hx
        for iy in range(max_iy + 1):
            keys.add((ix, iy))

    sites: dict[int, Site] = {}
    key_to_id: dict[tuple[int, int], int] = {}
    for site_id, (ix, iy) in enumerate(sorted(keys, key=lambda item: (item[1], item[0])), start=1):
        is_cross = ix % hx == 0 and iy % vy == 0
        site = Site(
            site_id=site_id,
            ix=ix,
            iy=iy,
            iz=0,
            x=config.box_padding + ix * config.s_spacing,
            y=config.box_padding + iy * config.s_spacing,
            z=z_plane,
            is_cross=is_cross,
        )
        sites[site_id] = site
        key_to_id[(ix, iy)] = site_id

    segments: list[Segment] = []
    for row in range(config.cells_y + 1):
        iy = row * vy
        for col in range(config.cells_x):
            start_ix = col * hx
            site_ids = tuple(key_to_id[(start_ix + offset, iy)] for offset in range(config.resolved_x_s_count))
            segments.append(Segment(f"H{row}_{col}", "horizontal", site_ids))

    for col in range(config.cells_x + 1):
        ix = col * hx
        for row in range(config.cells_y):
            start_iy = row * vy
            site_ids = tuple(key_to_id[(ix, start_iy + offset)] for offset in range(config.resolved_y_s_count))
            segments.append(Segment(f"V{col}_{row}", "vertical", site_ids))

    return sites, segments


def _build_cubic_grid(
    config: NetworkConfig,
    z_grid_layers: int,
    z_cell_count: int,
) -> tuple[dict[int, Site], list[Segment]]:
    hx = config.resolved_x_s_count - 1
    hy = config.resolved_y_s_count - 1
    hz = config.resolved_z_s_count - 1

    segment_keys: list[tuple[str, Orientation, tuple[tuple[int, int, int], ...]]] = []
    keys: set[tuple[int, int, int]] = set()

    for cell_x in range(config.cells_x):
        start_ix = cell_x * hx
        for grid_y in range(config.cells_y + 1):
            iy = grid_y * hy
            for grid_z in range(z_grid_layers):
                iz = grid_z * hz
                site_keys = tuple(
                    (start_ix + offset, iy, iz)
                    for offset in range(config.resolved_x_s_count)
                )
                keys.update(site_keys)
                segment_keys.append((f"X{cell_x}_{grid_y}_{grid_z}", "x", site_keys))

    for grid_x in range(config.cells_x + 1):
        ix = grid_x * hx
        for cell_y in range(config.cells_y):
            start_iy = cell_y * hy
            for grid_z in range(z_grid_layers):
                iz = grid_z * hz
                site_keys = tuple(
                    (ix, start_iy + offset, iz)
                    for offset in range(config.resolved_y_s_count)
                )
                keys.update(site_keys)
                segment_keys.append((f"Y{grid_x}_{cell_y}_{grid_z}", "y", site_keys))

    for grid_x in range(config.cells_x + 1):
        ix = grid_x * hx
        for grid_y in range(config.cells_y + 1):
            iy = grid_y * hy
            for cell_z in range(z_cell_count):
                start_iz = cell_z * hz
                site_keys = tuple(
                    (ix, iy, start_iz + offset)
                    for offset in range(config.resolved_z_s_count)
                )
                keys.update(site_keys)
                segment_keys.append((f"Z{grid_x}_{grid_y}_{cell_z}", "z", site_keys))

    sites: dict[int, Site] = {}
    key_to_id: dict[tuple[int, int, int], int] = {}
    for site_id, (ix, iy, iz) in enumerate(sorted(keys, key=lambda item: (item[2], item[1], item[0])), start=1):
        is_cross = ix % hx == 0 and iy % hy == 0 and iz % hz == 0
        site = Site(
            site_id=site_id,
            ix=ix,
            iy=iy,
            iz=iz,
            x=config.box_padding + ix * config.s_spacing,
            y=config.box_padding + iy * config.s_spacing,
            z=config.box_padding + iz * config.s_spacing,
            is_cross=is_cross,
        )
        sites[site_id] = site
        key_to_id[(ix, iy, iz)] = site_id

    segments = [
        Segment(segment_id, orientation, tuple(key_to_id[key] for key in site_keys))
        for segment_id, orientation, site_keys in segment_keys
    ]
    return sites, segments


def _select_insertions(
    config: NetworkConfig,
    segments: Iterable[Segment],
    cross_site_ids: frozenset[int],
) -> list[_InsertionPlan]:
    rng = random.Random(config.seed)
    if config.topology_mode == "2D_surface":
        axis_order: list[Orientation] = ["horizontal", "vertical"]
    elif config.has_z_segments:
        axis_order = ["x", "y", "z"]
    else:
        axis_order = ["x", "y"]

    eligible: dict[Orientation, list[Segment]] = {axis: [] for axis in axis_order}
    for segment in segments:
        if _eligible_starts(segment, cross_site_ids):
            eligible[segment.orientation].append(segment)

    for bucket in eligible.values():
        rng.shuffle(bucket)

    total_available = sum(len(bucket) for bucket in eligible.values())
    target_total = min(total_available, _round_half_up(config.insertion_density * total_available))
    targets = _allocate_insertions_by_axis(config, axis_order, eligible, target_total)

    selected_segments: list[Segment] = []
    for axis in axis_order:
        selected_segments.extend(eligible[axis][: targets[axis]])
    rng.shuffle(selected_segments)

    plans: list[_InsertionPlan] = []
    occupied: set[int] = set()
    for segment in selected_segments:
        starts = [
            start
            for start in _eligible_starts(segment, cross_site_ids)
            if not set(segment.site_ids[start : start + 3]).intersection(occupied)
        ]
        if not starts:
            continue
        start_index = rng.choice(starts)
        replaced = frozenset(segment.site_ids[start_index : start_index + 3])
        occupied.update(replaced)
        plans.append(
            _InsertionPlan(
                insertion_id=len(plans) + 1,
                segment=segment,
                start_index=start_index,
                replaced_site_ids=replaced,
                left_neighbor_site_id=segment.site_ids[start_index - 1],
                right_neighbor_site_id=segment.site_ids[start_index + 3],
            )
        )

    return sorted(plans, key=lambda plan: plan.insertion_id)


def _allocate_insertions_by_axis(
    config: NetworkConfig,
    axis_order: list[Orientation],
    eligible: dict[Orientation, list[Segment]],
    target_total: int,
) -> dict[Orientation, int]:
    ratios = config.normalized_axis_insertion_ratios
    weights = {axis: _axis_ratio_for_orientation(axis, ratios) for axis in axis_order}
    targets = {axis: 0 for axis in axis_order}
    if target_total <= 0:
        return targets

    if (
        axis_order == ["horizontal", "vertical"]
        and config.axis_insertion_weights is None
        and config.axis_insertion_ratios is None
    ):
        target_h = min(len(eligible["horizontal"]), _round_half_up(target_total * config.horizontal_ratio))
        target_v = min(len(eligible["vertical"]), target_total - target_h)
        remainder = target_total - target_h - target_v
        if remainder > 0:
            spare_h = len(eligible["horizontal"]) - target_h
            take_h = min(spare_h, remainder)
            target_h += take_h
            remainder -= take_h
        if remainder > 0:
            spare_v = len(eligible["vertical"]) - target_v
            target_v += min(spare_v, remainder)
        return {"horizontal": target_h, "vertical": target_v}

    raw_targets = {
        axis: target_total * weights[axis]
        for axis in axis_order
        if weights[axis] > 0.0 and eligible[axis]
    }
    for axis, raw_target in raw_targets.items():
        targets[axis] = min(len(eligible[axis]), math.floor(raw_target))

    assigned = sum(targets.values())
    remainder = target_total - assigned
    while remainder > 0:
        candidates = [
            axis
            for axis in raw_targets
            if targets[axis] < len(eligible[axis])
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda axis: (
                raw_targets[axis] - targets[axis],
                weights[axis],
                -axis_order.index(axis),
            ),
            reverse=True,
        )
        targets[candidates[0]] += 1
        remainder -= 1

    return targets


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _axis_ratio_for_orientation(
    orientation: Orientation,
    ratios: dict[str, float],
) -> float:
    if orientation in ("horizontal", "x"):
        return ratios["x"]
    if orientation in ("vertical", "y"):
        return ratios["y"]
    return ratios["z"]


def _eligible_starts(segment: Segment, cross_site_ids: frozenset[int]) -> list[int]:
    starts = []
    for start in range(1, len(segment.site_ids) - 3):
        replaced = set(segment.site_ids[start : start + 3])
        if replaced.isdisjoint(cross_site_ids):
            starts.append(start)
    return starts


def _build_bonds(
    segments: Iterable[Segment],
    replaced_site_ids: frozenset[int],
    site_atom_ids: dict[int, int],
    insertions: Iterable[Insertion],
) -> list[Bond]:
    bond_keys: set[tuple[int, int]] = set()
    raw_bonds: list[tuple[int, int, int]] = []

    for segment in segments:
        for left_site, right_site in zip(segment.site_ids, segment.site_ids[1:]):
            if left_site in replaced_site_ids or right_site in replaced_site_ids:
                continue
            atom1 = site_atom_ids[left_site]
            atom2 = site_atom_ids[right_site]
            key = _bond_key(atom1, atom2)
            if key not in bond_keys:
                bond_keys.add(key)
                raw_bonds.append((1, atom1, atom2))

    for insertion in insertions:
        for atom1, atom2 in (
            (insertion.left_neighbor_atom_id, insertion.left_anchor_atom_id),
            (insertion.right_anchor_atom_id, insertion.right_neighbor_atom_id),
        ):
            key = _bond_key(atom1, atom2)
            if key not in bond_keys:
                bond_keys.add(key)
                raw_bonds.append((2, atom1, atom2))

    return [
        Bond(bond_id=index, bond_type=bond_type, atom1=atom1, atom2=atom2)
        for index, (bond_type, atom1, atom2) in enumerate(raw_bonds, start=1)
    ]


def _axis(orientation: Orientation) -> tuple[float, float, float]:
    if orientation in ("horizontal", "x"):
        return (1.0, 0.0, 0.0)
    if orientation in ("vertical", "y"):
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _quaternion_for_orientation(orientation: Orientation) -> tuple[float, float, float, float]:
    sqrt_half = math.sqrt(0.5)
    if orientation in ("horizontal", "x"):
        return (sqrt_half, 0.0, sqrt_half, 0.0)
    if orientation in ("vertical", "y"):
        return (sqrt_half, -sqrt_half, 0.0, 0.0)
    return (1.0, 0.0, 0.0, 0.0)


def _compute_box_length(atoms: Iterable[Atom], padding: float) -> float:
    max_coord = 0.0
    for atom in atoms:
        max_coord = max(max_coord, atom.x, atom.y, atom.z)
    return max_coord + padding


def _bond_key(atom1: int, atom2: int) -> tuple[int, int]:
    return (atom1, atom2) if atom1 < atom2 else (atom2, atom1)


def _count_segments_by_axis(segments: Iterable[Segment]) -> dict[str, int]:
    counts = {"x": 0, "y": 0, "z": 0}
    for segment in segments:
        if segment.orientation in ("horizontal", "x"):
            counts["x"] += 1
        elif segment.orientation in ("vertical", "y"):
            counts["y"] += 1
        else:
            counts["z"] += 1
    return counts
