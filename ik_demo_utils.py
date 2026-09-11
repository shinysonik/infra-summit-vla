"""
Reusable IK-waypoint helpers for scripted demonstration generation.
Built and tested against the real model -- not estimated.
"""
import mink
import mujoco
import numpy as np


def solve_ik_to(configuration, frame_name, target_xyz, target_quat=None,
                 n_iter=300, dt=0.01, pos_cost=1.0, ori_cost=0.5):
    """Solve IK for one gripper to one target pose. Returns the resulting
    joint config (q) and the achieved position error in meters -- always
    check the error before trusting a waypoint, exactly like the reach-map
    test did."""
    task = mink.FrameTask(frame_name=frame_name, frame_type="body",
                           position_cost=pos_cost, orientation_cost=ori_cost if target_quat is not None else 0.0)
    if target_quat is not None:
        target = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_xyz)
    else:
        target = mink.SE3.from_translation(target_xyz)
    task.set_target(target)
    for _ in range(n_iter):
        vel = mink.solve_ik(configuration, [task], dt=dt, solver="daqp")
        configuration.integrate_inplace(vel, dt=dt)
    err = np.linalg.norm(task.compute_error(configuration)[:3])
    return configuration.q.copy(), err


def smooth_interp(q_start, q_end, n_steps):
    """Minimum-jerk interpolation (NOT linear) between two joint configs.
    Zero velocity/acceleration at both ends -- keeps the real simulated
    state tracking the commanded target instead of lagging behind it,
    which is what linear interpolation caused earlier in this project."""
    t = np.linspace(0, 1, n_steps)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5  # minimum-jerk profile
    return np.array([q_start + (q_end - q_start) * si for si in s])


def waypoint_sequence(configuration, model, waypoints, steps_per_segment=40):
    """waypoints: list of (frame_name, target_xyz, target_quat_or_None, gripper_value_or_None)
    Returns a list of full joint-angle arrays, one per simulation step,
    ready to feed as position-actuator targets."""
    qpos_trace = []
    gripper_trace = []
    q_current = configuration.q.copy()
    for frame_name, target_xyz, target_quat, gripper_val in waypoints:
        configuration.update(q_current)
        q_target, err = solve_ik_to(configuration, frame_name, target_xyz, target_quat)
        if err > 0.03:
            print(f"[ik_demo] WARNING: {frame_name} -> {target_xyz} converged to "
                  f"{err*100:.2f}cm error -- check this waypoint against the reach map")
        segment = smooth_interp(q_current, q_target, steps_per_segment)
        qpos_trace.extend(segment)
        gripper_trace.extend([gripper_val] * steps_per_segment if gripper_val is not None
                              else [None] * steps_per_segment)
        q_current = q_target
    return qpos_trace, gripper_trace
