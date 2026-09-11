"""
IK-waypoint helpers with dense frame-by-frame trajectory IK and continuous
collision avoidance against obstacles (e.g. bottle, bowl, plate, cup, drawer).
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
    # Obstacles the arm should route around. "drawer" covers the sliding body
    # and handle; "drawer_housing" covers the fixed back/side walls.
    tall_objects = ["bottle", "bowl", "plate", "cup", "drawer"]

    # The lid is visual-only (contype=0 in the MJCF) and the handle passes
    # through it -- if IK avoided it, the gripper could never reach the handle.
    # The drawer handle is the IK target itself and must not be an obstacle.
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

    # Cross-arm avoidance: left arm vs right arm.
    # Collect geoms by subtree (arm-base and everything below it). 5 cm
    # buffer matches mink's own dual-arm examples. Hard constraint, so if
    # the two arms need to share space, IK either finds a compromise or
    # raises NoSolutionFound -- never a silent physical collision.
    left_base_id = model.body("left_base").id
    right_base_id = model.body("right_base").id
    left_arm_geoms = mink.get_subtree_geom_ids(model, left_base_id)
    right_arm_geoms = mink.get_subtree_geom_ids(model, right_base_id)

    if left_arm_geoms and right_arm_geoms:
        cross_limit = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(left_arm_geoms, right_arm_geoms)],
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.10,
        )
        limits.append(cross_limit)

    return limits


def solve_ik_step(configuration, model, frame_name, target_xyz, target_quat=None,
                  limits=None, n_iter=8, dt=0.01, pos_cost=1.0, ori_cost=0.15,
                  frame_type="site"):
    """
    Single-arm IK step. frame_type defaults to "site" so frame_name is
    interpreted as a MuJoCo site (e.g. right_gripperframe) -- the actual
    fingertip TCP, not the wrist body origin. The site sits ~9.85cm from
    the body origin along local -Z, so aiming a body frame at a target
    makes the fingers stab ~10cm past it.
    """
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

    if limits is None:
        limits = get_avoidance_limits(model)

    for _ in range(n_iter):
        vel = mink.solve_ik(configuration, [task], dt=dt, solver="daqp", limits=limits)
        configuration.integrate_inplace(vel, dt=dt)

    err = np.linalg.norm(task.compute_error(configuration)[:3])
    return configuration.q.copy(), err


def waypoint_sequence(configuration, model, waypoints, steps_per_segment=20,
                      frame_type="site"):
    """
    Generates qpos trace by interpolating targets in Task Space and running IK
    with active CollisionAvoidanceLimit at EVERY frame step.

    frame_type defaults to "site" -- see solve_ik_step docstring.
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
                configuration,
                model,
                frame_name,
                curr_target_xyz,
                target_quat=target_quat,
                limits=limits,
                n_iter=8,
                dt=0.01,
                frame_type=frame_type,
            )
            qpos_trace.append(q_step)
            gripper_trace.append(gripper_val)

        current_poses[frame_name] = np.asarray(target_xyz).copy()

    return qpos_trace, gripper_trace