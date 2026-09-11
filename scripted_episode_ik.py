"""
Bimanual episode: right arm opens drawer AND holds it open, left arm
retrieves cutlery, right arm closes drawer, left arm picks plate.
"""
import numpy as np
import mink
import mujoco
from ik_demo_utils import waypoint_sequence

APPROACH_Z_OFFSET = 0.08
GRASP_OPEN = 1.2
GRASP_CLOSED = -0.15
CONTROL_DECIMATION = 10

# Calibrated handle grasp pose (site position + orientation), measured
# from a manual viewer grasp where the fingers wrap the handle.
HANDLE_POS = np.array([-0.01374231, 0.12252117, 0.91137901])
HANDLE_QUAT = np.array([0.04128911, -0.13647269, 0.68135373, 0.71793281])


def _replay_phase_on_scratch(model, initial_qpos, qpos_phase, grip_phase):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = initial_qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    for step in range(len(qpos_phase)):
        for act_id in range(model.nu):
            jid = model.actuator_trnid[act_id, 0]
            adr = model.jnt_qposadr[jid]
            scratch.ctrl[act_id] = qpos_phase[step][adr]
        gval = grip_phase[step]
        if gval is not None:
            side, val = gval
            scratch.ctrl[model.actuator(f"{side}_gripper").id] = val
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, scratch)

    return scratch


def build_drawer_and_cutlery_episode(model, data, randomizer):
    configuration = mink.Configuration(model)
    configuration.update(data.qpos.copy())

    handle_frame = "right_gripperframe"
    right_home_stage = np.array([0.22, 0.00, 0.90])

    slide_axis = model.jnt_axis[model.joint("drawer_slide").id]
    slide_range = model.jnt_range[model.joint("drawer_slide").id][1]
    pull_target = HANDLE_POS + slide_axis * slide_range

    # -------- PHASE 1: right arm opens drawer --------
    # 40 steps (was 20) so the drawer has time to follow the gripper. At 20
    # the gripper outruns the drawer's inertia and visually slides through
    # the handle. Right arm does NOT release at the end -- it holds the
    # handle through retrieve, or the drawer back-drives closed.
    open_waypoints = [
        (handle_frame, right_home_stage, None, GRASP_OPEN),
        (handle_frame, HANDLE_POS + [0, 0, APPROACH_Z_OFFSET], HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_CLOSED),
    ]
    qpos_open, grip_open = waypoint_sequence(
        configuration, model, open_waypoints, steps_per_segment=40
    )

    scratch = _replay_phase_on_scratch(model, data.qpos.copy(), qpos_open, grip_open)

    # -------- PHASE 2: LEFT retrieves; RIGHT holds handle open --------
    # Right arm is pinned at pull_target via a secondary FrameTask in every
    # solve -- it does not drift while the left arm moves.
    retrieve_waypoints = []
    for name in sorted(randomizer.last_in_drawer_items):
        cutlery_pos = scratch.xpos[model.body(name).id].copy()
        frame = "left_gripperframe"

        retrieve_waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos, None, GRASP_OPEN, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos, None, GRASP_CLOSED, "drawer"))
        retrieve_waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, "drawer"))

        z_height = float(cutlery_pos[2])
        # Place outside the open drawer's swept band so it doesn't get
        # swept closed by the drawer in the tail phase.
        place_pos = np.array([-0.15, -0.15, z_height])

        retrieve_waypoints.append((frame, place_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED))
        retrieve_waypoints.append((frame, place_pos, None, GRASP_CLOSED))
        retrieve_waypoints.append((frame, place_pos, None, GRASP_OPEN))

    qpos_retr, grip_retr = waypoint_sequence(
        configuration, model, retrieve_waypoints, steps_per_segment=20,
        secondary_frame="right_gripperframe",
        secondary_xyz=pull_target,
        secondary_quat=HANDLE_QUAT,
        secondary_pos_cost=1.0,
        secondary_ori_cost=0.15,
    )

    # -------- PHASE 3: right closes drawer, left picks plate --------
    tail_waypoints = [
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_CLOSED),  # push closed
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_OPEN),    # release
        (handle_frame, right_home_stage, None, GRASP_OPEN),     # retract
    ]

    plate_pos = data.xpos[model.body("plate").id].copy()
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