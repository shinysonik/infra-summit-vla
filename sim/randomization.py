"""
Domain randomization for the dinner-table scene.

Owned by: feature/mujoco-scene
Consumed by: /eval (calls reset(seed) once per episode, before stepping)

Design contract with randomization.yaml:
  - `apply_at: episode_reset` -> reset() is the ONLY place any of this runs.
  - Every randomization is relative to NOMINAL values captured from the
    compiled model at construction time, never cumulative. reset(seed=3)
    twice in a row must produce the identical scene both times.
  - Every axis named in the brief (object placement, weight, friction, shape,
    lighting, background) is sampled from a config range, never hardcoded.

Placement / collision avoidance
--------------------------------
Each object is placed independently via rejection sampling: sample a
candidate (x, y, yaw) from its own range in randomization.yaml, check it
against every already-placed object this reset, accept if the gap is >=
min_gap_between_objects_m, else resample (up to `max_attempts`, then fall
back to the object's nominal pose -- per randomization.yaml `placement`).

This is pure 2D geometry, no MuJoCo collision calls -- see the PRD/README for
why (short version: mj_collision needs a forward-kinematics pass per
candidate, which is unnecessary cost and a chicken-and-egg problem when
you're trying to decide the pose that produces that forward pass in the
first place; closed-form circle/segment distance is exact for these shapes
and effectively free).

Two footprint types, matching what the objects actually are:
  - circle:  plate, cup, bottle -- round, footprint = center + radius.
  - capsule: cutlery -- 12cm-long, 6mm-radius. Modeled as an actual line
    segment, NOT a bounding circle of radius (half_length + radius) -- a
    circle that size would be wildly conservative for how thin these
    objects really are and would make four of them nearly impossible to
    place close together. Segment endpoints are derived from the object's
    (x, y, yaw): the capsule geom lies along local Z by default and is
    rotated -90deg about X in the MJCF to lie flat along world Y, so an
    additional yaw rotation about world Z (the same composition
    _apply_pose uses) points its axis along (-sin(yaw), cos(yaw)).

Placement order: objects are placed largest-effective-radius-first (plate,
then cup/bottle, then cutlery). This isn't required for correctness -- it
reduces how often later, smaller objects have to retry, since the biggest
gaps get claimed by the object least able to route around a conflict.
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import mujoco
import yaml


@dataclass
class NominalState:
    body_pos: dict = field(default_factory=dict)
    body_quat: dict = field(default_factory=dict)
    body_mass: dict = field(default_factory=dict)
    geom_size: dict = field(default_factory=dict)
    geom_friction: dict = field(default_factory=dict)
    geom_rgba: dict = field(default_factory=dict)
    light_pos: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 2D collision geometry: circles and line segments, in table-local XY.
# --------------------------------------------------------------------------- #
def _capsule_endpoints(center_xy, yaw, half_length):
    """World-XY endpoints of a cutlery capsule at (x, y, yaw)."""
    direction = np.array([-math.sin(yaw), math.cos(yaw)])
    center = np.asarray(center_xy)
    return center - half_length * direction, center + half_length * direction


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _segment_segment_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4.
    Standard closest-point-between-two-segments algorithm
    (Ericson, Real-Time Collision Detection, ch. 5.1.9), clamped variant."""
    d1, d2, r = p2 - p1, p4 - p3, p1 - p3
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)

    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(p1 - p3))
    if a < 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        return float(np.linalg.norm((p3 + t * d2) - p1))

    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(-c / a, 0.0, 1.0)
        return float(np.linalg.norm(p1 + s * d1 - p3))

    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if abs(denom) > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


@dataclass
class Footprint:
    """A placed (or candidate) object's 2D collision shape."""
    shape: str            # "circle" or "capsule"
    center: np.ndarray
    radius: float
    p1: Optional[np.ndarray] = None   # capsule only
    p2: Optional[np.ndarray] = None   # capsule only


def _footprint_gap(a: Footprint, b: Footprint) -> float:
    """Surface-to-surface gap between two footprints (negative = overlapping)."""
    if a.shape == "circle" and b.shape == "circle":
        center_dist = float(np.linalg.norm(a.center - b.center))
    elif a.shape == "circle" and b.shape == "capsule":
        center_dist = _point_segment_dist(a.center, b.p1, b.p2)
    elif a.shape == "capsule" and b.shape == "circle":
        center_dist = _point_segment_dist(b.center, a.p1, a.p2)
    else:  # capsule-capsule
        center_dist = _segment_segment_dist(a.p1, a.p2, b.p1, b.p2)
    return center_dist - a.radius - b.radius


# --------------------------------------------------------------------------- #
class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, randomization_cfg_path: str):
        self.model = model
        with open(randomization_cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.seeds = self.cfg["seeds"]
        self.ranges = self.cfg["ranges"]
        self.object_cfg = self.cfg["objects"]
        self.placement_cfg = self.cfg["placement"]
        self.object_names = list(self.object_cfg.keys())
        self._capture_nominal()

    # ------------------------------------------------------------------ #
    def _capture_nominal(self):
        m = self.model
        self.nominal = NominalState()

        for name in self.object_names:
            bid = m.body(name).id
            jnt_adr = m.body_jntadr[bid]
            assert m.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE, (
                f"{name} must have a free joint for pose randomization"
            )
            qpos_adr = m.jnt_qposadr[jnt_adr]
            self.nominal.body_pos[name] = m.qpos0[qpos_adr:qpos_adr + 3].copy()
            self.nominal.body_quat[name] = m.qpos0[qpos_adr + 3:qpos_adr + 7].copy()
            self.nominal.body_mass[name] = float(m.body_mass[bid])

            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == bid:
                    self.nominal.geom_size[gid] = m.geom_size[gid].copy()
                    self.nominal.geom_friction[gid] = m.geom_friction[gid].copy()

        table_bid = m.body("table").id
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] == table_bid:
                self.nominal.geom_rgba[gid] = m.geom_rgba[gid].copy()

        for lid in range(m.nlight):
            self.nominal.light_pos[lid] = m.light_pos[lid].copy()

    # ------------------------------------------------------------------ #
    def reset(self, data: mujoco.MjData, seed: int):
        if seed not in self.seeds:
            raise ValueError(
                f"seed {seed} not in randomization.yaml seeds list {self.seeds}"
            )
        rng = np.random.RandomState(seed)
        mujoco.mj_resetData(self.model, data)

        self._randomize_object_placement(data, rng)
        self._randomize_mass(rng)
        self._randomize_friction(rng)
        self._randomize_shape(rng)
        self._randomize_lighting(rng)
        self._randomize_background(rng)

        mujoco.mj_forward(self.model, data)

    # ------------------------------------------------------------------ #
    def _make_footprint(self, name, x, y, yaw) -> Footprint:
        cfg = self.object_cfg[name]
        center = np.array([x, y])
        if cfg["shape"] == "circle":
            return Footprint(shape="circle", center=center, radius=cfg["radius_m"])
        p1, p2 = _capsule_endpoints(center, yaw, cfg["half_length_m"])
        return Footprint(shape="capsule", center=center, radius=cfg["radius_m"], p1=p1, p2=p2)

    def _violates_drawer_avoidance(self, name, footprint: Footprint) -> bool:
        cfg = self.object_cfg[name]
        if cfg["shape"] != "capsule":
            return False  # soft rule applies to cutlery only, per config docstring
        da = self.placement_cfg.get("drawer_avoidance")
        if not da:
            return False
        drawer_y_min = da["drawer_y_center_m"] - da["drawer_half_extent_y_m"] - da["extra_margin_m"]
        # Check the capsule's endpoints, not just its center -- a segment can
        # poke into the exclusion zone even if its center doesn't.
        return footprint.p1[1] > drawer_y_min or footprint.p2[1] > drawer_y_min

    def _randomize_object_placement(self, data, rng):
        max_attempts = self.placement_cfg["max_attempts"]
        min_gap = self.placement_cfg["min_gap_between_objects_m"]

        # Largest effective radius first, so big objects (hardest to route
        # around) claim space before small ones have to dodge them.
        def effective_radius(name):
            cfg = self.object_cfg[name]
            return cfg["radius_m"] + (cfg["half_length_m"] if cfg["shape"] == "capsule" else 0.0)

        order = sorted(self.object_names, key=effective_radius, reverse=True)

        placed: list[Footprint] = []
        fallback_count = 0

        for name in order:
            cfg = self.object_cfg[name]
            x_lo, x_hi = cfg["x"]
            y_lo, y_hi = cfg["y"]
            yaw_lo, yaw_hi = cfg["yaw"]

            accepted = None
            for _ in range(max_attempts):
                x = rng.uniform(x_lo, x_hi)
                y = rng.uniform(y_lo, y_hi)
                yaw = rng.uniform(yaw_lo, yaw_hi)
                fp = self._make_footprint(name, x, y, yaw)

                if self._violates_drawer_avoidance(name, fp):
                    continue
                if any(_footprint_gap(fp, other) < min_gap for other in placed):
                    continue

                accepted = (x, y, yaw, fp)
                break

            if accepted is None:
                # Fall back to nominal pose (dx=dy=dyaw=0), per randomization.yaml.
                fallback_count += 1
                nominal_pos = self.nominal.body_pos[name]
                x, y, yaw = float(nominal_pos[0]), float(nominal_pos[1]), 0.0
                fp = self._make_footprint(name, x, y, yaw)
                accepted = (x, y, yaw, fp)

            x, y, yaw, fp = accepted
            placed.append(fp)
            self._apply_pose(data, name, x, y, yaw)

        if fallback_count:
            # Not an error -- the system is designed to degrade gracefully --
            # but worth surfacing if it happens often, since it means the
            # ranges/margins are tight relative to the number of objects.
            print(f"[randomization] seed used {fallback_count} nominal-pose "
                  f"fallback(s) after {max_attempts} placement attempts each")

    def _apply_pose(self, data, name, x, y, dyaw):
        bid = self.model.body(name).id
        jnt_adr = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]

        nominal_pos = self.nominal.body_pos[name]
        nominal_quat = self.nominal.body_quat[name]

        new_pos = np.array([x, y, nominal_pos[2]])

        yaw_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), dyaw)
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, yaw_quat, nominal_quat)

        data.qpos[qpos_adr:qpos_adr + 3] = new_pos
        data.qpos[qpos_adr + 3:qpos_adr + 7] = new_quat
        dof_adr = self.model.jnt_dofadr[jnt_adr]
        data.qvel[dof_adr:dof_adr + 6] = 0.0

    # ------------------------------------------------------------------ #
    # Non-spatial randomization axes -- unchanged in behavior from before.
    # ------------------------------------------------------------------ #
    def _randomize_mass(self, rng):
        lo, hi = self.ranges["mass_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            self.model.body_mass[bid] = self.nominal.body_mass[name] * scale

    def _randomize_friction(self, rng):
        lo, hi = self.ranges["friction_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            for gid, nominal_fric in self.nominal.geom_friction.items():
                if self.model.geom_bodyid[gid] == bid:
                    self.model.geom_friction[gid] = nominal_fric * scale

    def _randomize_shape(self, rng):
        lo, hi = self.ranges["shape_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            for gid, nominal_size in self.nominal.geom_size.items():
                if self.model.geom_bodyid[gid] == bid:
                    self.model.geom_size[gid] = nominal_size * scale

    def _randomize_lighting(self, rng):
        amb_lo, amb_hi = self.ranges["lighting"]["ambient"]
        dif_lo, dif_hi = self.ranges["lighting"]["diffuse"]
        jit_lo, jit_hi = self.ranges["lighting"]["position_jitter_m"]

        for lid in range(self.model.nlight):
            amb = rng.uniform(amb_lo, amb_hi)
            dif = rng.uniform(dif_lo, dif_hi)
            jitter = rng.uniform(jit_lo, jit_hi, size=3)

            self.model.light_ambient[lid] = np.array([amb, amb, amb])
            self.model.light_diffuse[lid] = np.array([dif, dif, dif])
            self.model.light_pos[lid] = self.nominal.light_pos[lid] + jitter

    def _randomize_background(self, rng):
        # PLACEHOLDER: no real background texture set exists on disk yet.
        # Approximated as a table color tint until real textures land.
        if not self.ranges["background"]["randomize_table_material"]:
            return
        for gid, nominal_rgba in self.nominal.geom_rgba.items():
            tint = rng.uniform(0.7, 1.0, size=3)
            new_rgba = nominal_rgba.copy()
            new_rgba[:3] = nominal_rgba[:3] * tint
            self.model.geom_rgba[gid] = new_rgba
