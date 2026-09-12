"""
IK-waypoint helpers with dense frame-by-frame trajectory IK and continuous
collision avoidance against obstacles.
"""
import mink
import mujoco
import numpy as np


def actuator_qpos_addresses(model):
    addrs = []
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        addrs.append(int(model.jnt_qposadr[joint_id]))
    return addrs


def _geoms_by_body_prefix(model, prefix):
    gids = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(prefix):
            gids.append(gid)
    return gids


def get_avoidance_limits(model, exclude_body=None):
    tall_objects = ["bottle", "bowl", "plate", "cup", "drawer"]
    skip_geoms = {"drawer_handle"}

    obj_geoms = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname in skip_geoms:
            continue
        if (bname in tall_objects or any(x in gname for x in tall_objects)) and bname != exclude_body:
            obj_geoms.append(gid)

    left_geoms = _geoms_by_body_prefix(model, "left_")
    right_geoms = _geoms_by_body_prefix(model, "right_")

    limits = []
    for arm_geoms in (left_geoms, right_geoms):
        if not arm_geoms or not obj_geoms:
            continue
        lim = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(arm_geoms, obj_geoms)],
            minimum_distance_from_collisions=0.015,
            collision_detection_distance=0.05,
        )
        limits.append(lim)

    left_base_id = model.body("left_base").id
    right_base_id = model.body("right_base").id
    left_arm_geoms = mink.get_subtree_geom_ids(model, left_base_id)
    right_arm_geoms = mink.get_subtree_geom_ids(model, right_base_id)
    if left_arm_geoms and right_arm_geoms:
        cross_limit = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(left_arm_geoms, right_arm_geoms)],
            minimum_distance_from_collisions=0.03,
            collision_detection_distance=0.06,
        )
        limits.append(cross_limit)
        limits.append(mink.ConfigurationLimit(model))

    return limits


def joint_space_lerp(configuration, model, joint_names, target_values, steps=30):
    """Linearly interpolate specific joints from their CURRENT configuration
    values to target_values, holding every other DOF fixed. Bypasses mink's
    Cartesian IK (and its nullspace ambiguity) entirely for this segment --
    use it to route through a manually-verified, known-collision-free
    configuration instead of letting the solver pick an arbitrary elbow/wrist
    branch on its own.

    NOTE: this does not run CollisionAvoidanceLimit -- it trusts that the
    straight joint-space line from the current (already-safe) pose to the
    target (manually verified) pose stays safe. Verify with a contact check
    on your real meshes before trusting it in production.
    """
    q_start = configuration.q.copy()
    adrs = [model.jnt_qposadr[model.joint(jn).id] for jn in joint_names]
    start_vals = [q_start[adr] for adr in adrs]

    qpos_trace = []
    for i in range(steps):
        s = (i + 1) / steps
        q = q_start.copy() if i == 0 else qpos_trace[-1].copy()
        for adr, sv, tv in zip(adrs, start_vals, target_values):
            q[adr] = (1.0 - s) * sv + s * tv
        qpos_trace.append(q)

    configuration.update(qpos_trace[-1])
    return qpos_trace


def compute_gripper_closing_axis_local(model, data, gripper_body, moving_jaw_body, gripper_joint,
                                        reference_qpos_5, arm_joint_names, dtheta=0.02):
    """Measure the gripper's closing-motion direction, expressed in the
    gripper site's own local frame. This is a MECHANISM CONSTANT: it does
    not depend on the arm's shoulder/elbow/wrist_roll configuration (verified
    empirically -- recomputing it from two unrelated arm poses gives the same
    vector to ~1e-14). Only recompute this if the gripper geometry itself
    changes (different STL, different joint placement).

    Returns the unit vector in the `<gripper_body>frame` site's local frame.
    """
    for jn, val in zip(arm_joint_names, reference_qpos_5):
        data.qpos[model.jnt_qposadr[model.joint(jn).id]] = val
    g_adr = model.jnt_qposadr[model.joint(gripper_joint).id]
    data.qpos[g_adr] = 0.3
    mujoco.mj_forward(model, data)

    site_id = model.site(f"{gripper_body}frame").id
    tcp_world = data.site_xpos[site_id].copy()
    site_R = data.site_xmat[site_id].reshape(3, 3).copy()

    jaw_bid = model.body(moving_jaw_body).id
    p = data.xpos[jaw_bid]
    R = data.xmat[jaw_bid].reshape(3, 3)
    local_on_jaw = R.T @ (tcp_world - p)

    def world_from_local(local_pt):
        p = data.xpos[jaw_bid]
        R = data.xmat[jaw_bid].reshape(3, 3)
        return p + R @ local_pt

    data.qpos[g_adr] = 0.3 + dtheta
    mujoco.mj_forward(model, data)
    p_plus = world_from_local(local_on_jaw)
    data.qpos[g_adr] = 0.3 - dtheta
    mujoco.mj_forward(model, data)
    p_minus = world_from_local(local_on_jaw)

    tangent_world = p_plus - p_minus
    tangent_world /= np.linalg.norm(tangent_world)
    return site_R.T @ tangent_world


def _rotation_from_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def compute_plate_rim_grasp(model, data, configuration, plate_center_xy, left_base_xy,
                             plate_radius, plate_z, closing_axis_local, arm_joint_names,
                             gripper_frame_name, home_stage_xyz, rim_fraction=0.9,
                             outside_margin=0.06, sweep_steps_per_segment=15, phi_step_deg=15,
                             grasp_side="near", max_orientation_error_deg=25.0):
    """Find the plate-rim grasp point + orientation for THIS episode's actual
    plate position. Pinches the rim vertically (closing_axis_local -> world Z)
    so the jaws squeeze the plate's top/bottom surface at the rim (thin, ~1.5cm,
    well within the gripper's ~4cm mouth) instead of the flat center (which
    slips under load).

    IMPORTANT, learned the hard way on the real robot mesh: the left arm is
    only 5-DOF (5 revolute joints + gripper). A fully-specified 6D target
    (3 position + 3 orientation) generally has NO exact solution for a 5-DOF
    chain. mink's FrameTask resolves this as a weighted least-squares
    compromise, and which compromise it lands on is EXTREMELY sensitive to
    step resolution -- a coarse check (sweep_steps_per_segment=6, an earlier
    version of this function) can report a converged, well-margined solution
    that a full-resolution replay (steps=15-30) reveals is actually a false
    positive: same quat, position error balloons from ~1cm to ~9cm because
    the coarse path skipped past the true (bad) equilibrium the fine path
    correctly settles into. Verified directly on this rig: steps=4/6/8 gave
    err < 5cm, steps=10 gave 8.6cm, steps>=15 all converged to the SAME
    9.4cm error (a real, reproducible attractor, not noise or slow
    convergence -- 50 steps gives an identical result to 15).

    Consequently this function ranks candidates on BOTH the achieved
    position error AND the achieved orientation error (angle between the
    resulting closing axis and world vertical) -- NOT on joint-limit margin,
    which does not correlate with whether the least-squares compromise is
    any good. And it evaluates every candidate at close to full resolution
    (sweep_steps_per_segment=15 default) because coarser checks are not a
    reliable proxy for the fine-resolution outcome, as shown above -- this
    makes the sweep slower (~30s for 48 candidates on the real meshes) but
    the 6-step version's speed was not real, it was reporting wrong answers
    fast.

    max_orientation_error_deg: candidates whose achieved closing axis is
    more than this many degrees from vertical are rejected even if their
    position error is excellent -- a badly tilted closing axis will not
    pinch the rim top/bottom the way this grasp is designed to. On the one
    real seed tested so far the best achievable compromise was ~20 degrees
    off vertical, not 0 -- this is a real, seed-dependent kinematic limit
    of the 5-DOF arm, not a bug; if every candidate gets rejected, that
    seed's plate position may need `grasp_side="far"` or a smaller
    `rim_fraction` (closer to center, shorter reach) instead.
    """
    to_base = left_base_xy - plate_center_xy
    to_base_dir = to_base / np.linalg.norm(to_base)
    if grasp_side == "far":
        to_base_dir = -to_base_dir
    elif grasp_side != "near":
        raise ValueError(f"grasp_side must be 'near' or 'far', got {grasp_side!r}")
    rim_xy = plate_center_xy + rim_fraction * plate_radius * to_base_dir
    rim_xyz = np.array([rim_xy[0], rim_xy[1], plate_z])
    outside_xyz = rim_xyz + outside_margin * np.array([to_base_dir[0], to_base_dir[1], 0.0])

    q0 = configuration.q.copy()
    best = None
    for e1_sign in (1.0, -1.0):
        R0 = _rotation_from_a_to_b(closing_axis_local, np.array([0.0, 0.0, e1_sign]))
        for phi_deg in range(0, 360, phi_step_deg):
            phi = np.radians(phi_deg)
            Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])
            R = Rz @ R0
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R.flatten())

            cfg = mink.Configuration(model)
            cfg.update(q0)
            waypoints = [
                (gripper_frame_name, home_stage_xyz, None, 1.2, None),
                (gripper_frame_name, outside_xyz, quat, 1.2, "plate"),
                (gripper_frame_name, rim_xyz, quat, 1.2, "plate"),
            ]
            try:
                waypoint_sequence(cfg, model, waypoints, steps_per_segment=sweep_steps_per_segment)
            except Exception:
                continue
            transform = cfg.get_transform_frame_to_world(gripper_frame_name, "site")
            final_site = transform.translation()
            final_R = transform.rotation().as_matrix()
            pos_err_cm = np.linalg.norm(final_site - rim_xyz) * 100
            closing_world = final_R @ closing_axis_local
            angle_from_vertical_deg = np.degrees(np.arccos(np.clip(abs(closing_world[2]), 0, 1)))

            if angle_from_vertical_deg > max_orientation_error_deg:
                continue
            score = pos_err_cm + 0.3 * angle_from_vertical_deg
            if best is None or score < best[0]:
                best = (score, pos_err_cm, angle_from_vertical_deg, quat.copy())

    if best is None:
        raise RuntimeError(
            "compute_plate_rim_grasp: no candidate satisfied max_orientation_error_deg "
            f"({max_orientation_error_deg}) for this plate position. Try grasp_side='far', "
            "a smaller rim_fraction, or relax max_orientation_error_deg."
        )
    _, pos_err_cm, angle_from_vertical_deg, quat = best

    # IMPORTANT: scripted_episode_ik.py's actual tail waypoints pass
    # target_quat=None (not the quat returned here) -- an earlier attempt to
    # use it directly made the 5-DOF arm sacrifice position for orientation
    # in at least one case (~8cm miss). Because of that, the sweep above
    # (which DOES use quat to pick a good orientation) does not necessarily
    # predict what the real, orientation-free execution will achieve. Found
    # directly on seed 10012 (plate at world x=+0.25, far from the left base
    # at x=-0.22, likely at or past this arm's reach limit): the sweep
    # reported 4.5cm error, but replaying the REAL waypoints (target_quat=
    # None) gave 13.4cm -- the arm never actually touched the plate, the
    # weld never activated, and the "episode" silently did nothing.
    #
    # So: re-verify with the SAME target_quat=None path scripted_episode_ik.py
    # actually uses, and return THAT number. This is the trustworthy signal
    # for whether this specific (randomized) plate position is reachable at
    # all -- use it to skip/flag episodes the same way SKIP_RETRIEVE already
    # does for cutlery, rather than silently shipping a "successful-looking"
    # trajectory where the gripper never touched anything.
    verify_cfg = mink.Configuration(model)
    verify_cfg.update(q0)
    verify_waypoints = [
        (gripper_frame_name, home_stage_xyz, None, 1.2, None),
        (gripper_frame_name, outside_xyz, None, 1.2, "plate"),
        (gripper_frame_name, rim_xyz, None, 1.2, "plate"),
    ]
    waypoint_sequence(verify_cfg, model, verify_waypoints, steps_per_segment=20)
    verify_final = verify_cfg.get_transform_frame_to_world(gripper_frame_name, "site").translation()
    true_pos_err_cm = float(np.linalg.norm(verify_final - rim_xyz) * 100)

    return rim_xyz, outside_xyz, quat, true_pos_err_cm, angle_from_vertical_deg


def activate_grasp_connect(model, data, eq_name, body1_name, body2_name, world_anchor_pt):
    """Activate a `connect` equality constraint, anchored at the CURRENT physical
    grasp point. Must be called AFTER the gripper has physically closed (contact
    settled) and BEFORE any pulling motion. Call deactivate_grasp_connect to release.

    This does not require correct mesh collision geometry -- it directly enforces
    "these two bodies stay joined at this point," which is what a closed gripper
    with sufficient friction is physically doing anyway. It replaces reliance on
    the moving jaw's convex-hull mesh contact for load-bearing grip.
    """
    eq_id = model.equality(eq_name).id
    b1 = model.body(body1_name).id
    b2 = model.body(body2_name).id

    def world_to_local(bid, world_pt):
        p = data.xpos[bid]
        R = data.xmat[bid].reshape(3, 3)
        return R.T @ (world_pt - p)

    model.eq_data[eq_id, 0:3] = world_to_local(b1, world_anchor_pt)
    model.eq_data[eq_id, 3:6] = world_to_local(b2, world_anchor_pt)
    data.eq_active[eq_id] = 1
    model.eq_active0[eq_id] = 1


def deactivate_grasp_connect(model, data, eq_name):
    eq_id = model.equality(eq_name).id
    data.eq_active[eq_id] = 0
    model.eq_active0[eq_id] = 0


def activate_grasp_weld(model, data, eq_name, body1_name, body2_name):
    """Weld body2 rigidly to body1 at their CURRENT relative pose (6-DOF lock
    -- translation + orientation). Use for objects that must not rotate
    relative to the grasping hand once gripped (a free-floating plate on a
    connect/3-DOF-point constraint will swing like a pendulum around the
    anchor as the arm moves -- verified, ~7cm lateral drift over an 8cm lift).

    body1_name MUST be a body that does NOT itself rotate as the gripper
    hinge closes -- i.e. the FIXED wrist structure (e.g. "left_gripper"),
    NOT the moving jaw (e.g. "left_moving_jaw_so101_v1"). If you weld to the
    moving jaw and activate before the gripper has finished closing, the
    welded object is rigidly attached to a frame that keeps rotating for
    the remainder of the close motion, and gets flung with it. Verified
    directly: welding to the moving jaw mid-close gives ~21 degrees of
    rotation drift and ~3.4cm position drift by the time the gripper
    finishes closing; welding to the fixed wrist body under the exact same
    conditions gives ~1cm / ~0.5 degrees once settled -- delaying activation
    until the gripper is fully closed does NOT fix this on its own (tested:
    0.2 degree difference, i.e. no meaningful effect) if you're still
    welding to the moving jaw. The body choice is what matters, not timing.

    eq_data layout for MuJoCo's <weld>, confirmed empirically on this
    MuJoCo version (not assumed): eq_data[0:3] = anchor (body1 frame),
    eq_data[3:6] = relpose position (body1 frame), eq_data[6:10] = relpose
    quaternion wxyz, eq_data[10] = torquescale (default 1.0, left
    unmodified here).
    """
    eq_id = model.equality(eq_name).id
    b1 = model.body(body1_name).id
    b2 = model.body(body2_name).id
    p1 = data.xpos[b1].copy()
    R1 = data.xmat[b1].reshape(3, 3)
    p2 = data.xpos[b2].copy()
    R2 = data.xmat[b2].reshape(3, 3)
    rel_pos = R1.T @ (p2 - p1)
    R_rel = R1.T @ R2
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, R_rel.flatten())
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1
    model.eq_active0[eq_id] = 1


def deactivate_grasp_weld(model, data, eq_name):
    eq_id = model.equality(eq_name).id
    data.eq_active[eq_id] = 0
    model.eq_active0[eq_id] = 0


def solve_ik_step(configuration, model, frame_name, target_xyz, target_quat=None,
                  limits=None, n_iter=8, dt=0.01, pos_cost=1.0, ori_cost=0.15,
                  frame_type="site",
                  secondary_frame=None, secondary_xyz=None, secondary_quat=None,
                  secondary_pos_cost=1.0, secondary_ori_cost=0.15):
    """Single IK step. If secondary_* is given, a second FrameTask pins
    that frame at the secondary target in every solve -- used for bimanual
    ops where one arm moves and the other must hold its pose."""
    task = mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_cost=pos_cost,
        orientation_cost=ori_cost if target_quat is not None else 0.0,
    )
    if target_quat is not None:
        target = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_xyz)
    else:
        target = mink.SE3.from_translation(target_xyz)
    task.set_target(target)

    tasks = [task]
    if secondary_frame is not None and secondary_xyz is not None:
        sec_task = mink.FrameTask(
            frame_name=secondary_frame,
            frame_type=frame_type,
            position_cost=secondary_pos_cost,
            orientation_cost=secondary_ori_cost if secondary_quat is not None else 0.0,
        )
        if secondary_quat is not None:
            sec_target = mink.SE3.from_rotation_and_translation(
                mink.SO3(secondary_quat), secondary_xyz)
        else:
            sec_target = mink.SE3.from_translation(secondary_xyz)
        sec_task.set_target(sec_target)
        tasks.append(sec_task)

    if limits is None:
        limits = get_avoidance_limits(model)

    for _ in range(n_iter):
        vel = mink.solve_ik(configuration, tasks, dt=dt, solver="daqp", limits=limits)
        configuration.integrate_inplace(vel, dt=dt)

    err = np.linalg.norm(task.compute_error(configuration)[:3])
    return configuration.q.copy(), err


def waypoint_sequence(configuration, model, waypoints, steps_per_segment=20,
                      frame_type="site",
                      secondary_frame=None, secondary_xyz=None, secondary_quat=None,
                      secondary_pos_cost=1.0, secondary_ori_cost=0.15):
    """
    Dense frame-by-frame IK over a waypoint list, min-jerk interpolation,
    CollisionAvoidanceLimit at every step.

    gripper_trace entries are ("left"|"right", value) tuples or None.
    Side is inferred from the waypoint's frame name -- avoids the old bug
    where every gripper_trace value was applied to right_gripper only,
    silently leaving the left gripper uncommanded.
    """
    qpos_trace = []
    gripper_trace = []
    current_poses = {}

    for wp in waypoints:
        frame_name = wp[0]
        target_xyz = wp[1]
        target_quat = wp[2]
        gripper_val = wp[3]
        exclude_body = wp[4] if len(wp) > 4 else None

        if frame_name not in current_poses:
            current_poses[frame_name] = configuration.get_transform_frame_to_world(
                frame_name, frame_type
            ).translation().copy()

        start_xyz = current_poses[frame_name]
        limits = get_avoidance_limits(model, exclude_body=exclude_body)

        t_vals = np.linspace(0, 1, steps_per_segment)
        s_vals = 10 * t_vals**3 - 15 * t_vals**4 + 6 * t_vals**5

        for s in s_vals:
            curr_target_xyz = (1.0 - s) * start_xyz + s * np.asarray(target_xyz)

            q_step, err = solve_ik_step(
                configuration, model, frame_name, curr_target_xyz,
                target_quat=target_quat, limits=limits,
                n_iter=8, dt=0.01, frame_type=frame_type,
                secondary_frame=secondary_frame,
                secondary_xyz=secondary_xyz,
                secondary_quat=secondary_quat,
                secondary_pos_cost=secondary_pos_cost,
                secondary_ori_cost=secondary_ori_cost,
            )
            qpos_trace.append(q_step)

            if gripper_val is None:
                gripper_trace.append(None)
            elif "left" in frame_name:
                gripper_trace.append(("left", gripper_val))
            elif "right" in frame_name:
                gripper_trace.append(("right", gripper_val))
            else:
                gripper_trace.append(None)

        current_poses[frame_name] = np.asarray(target_xyz).copy()

    return qpos_trace, gripper_trace