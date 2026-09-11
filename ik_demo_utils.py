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

    return limits


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