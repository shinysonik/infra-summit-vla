"""
Builds an IK trajectory for one episode: open drawer, retrieve cutlery,
close drawer, then move a plate. Arms are hardcoded per grasp.
"""
import numpy as np
import mink
import mujoco
from ik_demo_utils import waypoint_sequence

APPROACH_Z_OFFSET = 0.08
GRASP_OPEN = 1.2
# Gripper hinge range is [-0.174533, 1.74533]. 0.0 is mid-range -- fingers
# barely touch. -0.15 is near-closed and actually squeezes the handle.
GRASP_CLOSED = -0.15

# Must match control_decimation used by the caller when replaying the trace.
CONTROL_DECIMATION = 10


def _replay_phase_on_scratch(model, initial_qpos, qpos_phase, grip_phase):
    """Replay a joint-space trajectory on a scratch MjData and return it.

    Used to find out where free-joint objects ACTUALLY end up after a phase,
    rather than predicting it analytically. Cutlery inside the drawer rides
    with the drawer's back wall when it opens -- displacement depends on
    friction, yaw, and mass, so it is not equal to slide_range in general
    (measured: 14.5cm on fork_1 for a 16cm slide, plus ~1.5cm lateral drift).
    Simulation is the only correct source of truth here.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = initial_qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    for step in range(len(qpos_phase)):
        for act_id in range(model.nu):
            jid = model.actuator_trnid[act_id, 0]
            adr = model.jnt_qposadr[jid]
            scratch.ctrl[act_id] = qpos_phase[step][adr]
        if grip_phase[step] is not None:
            scratch.ctrl[model.actuator("right_gripper").id] = grip_phase[step]
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, scratch)

    return scratch


def build_drawer_and_cutlery_episode(model, data, randomizer):
    configuration = mink.Configuration(model)
    configuration.update(data.qpos.copy())

    # Calibrated handle grasp pose: site position and orientation measured
    # from a manual grasp in the viewer where the fingers wrap the handle.
    # The drawer/handle are NOT randomized (fixed at (0, 0.24, 0.81)), so
    # this pose is valid for every seed. Recorded interactively -- this is
    # not a theoretical quat, it is an actually-reachable SO-101 pose.
    handle_pos = np.array([-0.01374231, 0.12252117, 0.91137901])
    HANDLE_QUAT = np.array([0.04128911, -0.13647269, 0.68135373, 0.71793281])
    handle_frame = "right_gripperframe"

    # Staging pose for the right arm before approaching the drawer.
    right_home_stage = np.array([0.22, 0.00, 0.90])

    slide_axis = model.jnt_axis[model.joint("drawer_slide").id]
    slide_range = model.jnt_range[model.joint("drawer_slide").id][1]
    pull_target = handle_pos + slide_axis * slide_range

    # -------- PHASE 1: open drawer --------
    open_waypoints = [
        (handle_frame, right_home_stage, None, GRASP_OPEN),
        (handle_frame, handle_pos + [0, 0, APPROACH_Z_OFFSET], HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, handle_pos, HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, handle_pos, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_OPEN),
    ]
    qpos_open, grip_open = waypoint_sequence(
        configuration, model, open_waypoints, steps_per_segment=20
    )

    # Replay open phase on a scratch sim to learn the real cutlery poses.
    # Without this, retrieve planning targets the reset pose (drawer closed,
    # cutlery 14-16cm behind where it actually is), IK cannot reach, and the
    # arm just spins the wrist in place.
    scratch = _replay_phase_on_scratch(model, data.qpos.copy(), qpos_open, grip_open)

    # -------- PHASE 2: retrieve cutlery (from real post-open positions) --------
    retrieve_waypoints = []
    for name in sorted(randomizer.last_in_drawer_items):
        cutlery_pos = scratch.xpos[model.body(name).id].copy()
        frame = "right_gripperframe"

        # exclude_body="drawer" lets the gripper dive inside the open drawer
        # without IK's CollisionAvoidanceLimit pushing it out of the cavity.
        retrieve_waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos, None, GRASP_OPEN, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos, None, GRASP_CLOSED, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, "drawer"))

        z_height = float(cutlery_pos[2])
        place_pos = np.array([0.0, 0.10, z_height])

        retrieve_waypoints.append((frame, place_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED))
        retrieve_waypoints.append((frame, place_pos, None, GRASP_CLOSED))
        retrieve_waypoints.append((frame, place_pos, None, GRASP_OPEN))

    qpos_retr, grip_retr = waypoint_sequence(
        configuration, model, retrieve_waypoints, steps_per_segment=20
    )

    # -------- PHASE 3: close drawer + pick plate --------
    # Same handle grasp pose for closing as for opening.
    tail_waypoints = [
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, handle_pos, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, handle_pos, HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, right_home_stage, None, GRASP_OPEN),
    ]

    plate_pos = data.xpos[model.body("plate").id].copy()
    # Same site-vs-body issue on the left arm: use the fingertip site.
    plate_arm = "left_gripperframe"
    left_home_stage = np.array([-0.22, 0.00, 0.90])

    tail_waypoints.append((plate_arm, left_home_stage, None, GRASP_OPEN))
    tail_waypoints.append((plate_arm, plate_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN, "plate"))
    tail_waypoints.append((plate_arm, plate_pos, None, GRASP_OPEN, "plate"))
    tail_waypoints.append((plate_arm, plate_pos, None, GRASP_CLOSED, "plate"))
    tail_waypoints.append((plate_arm, plate_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, "plate"))

    qpos_tail, grip_tail = waypoint_sequence(
        configuration, model, tail_waypoints, steps_per_segment=20
    )

    qpos_trace = qpos_open + qpos_retr + qpos_tail
    gripper_trace = grip_open + grip_retr + grip_tail
    return qpos_trace, gripper_trace