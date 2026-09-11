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