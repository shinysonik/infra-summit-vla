"""
Bimanual episode: right arm opens drawer AND holds it open, left arm
retrieves cutlery, right arm closes drawer, left arm picks plate.
"""
import numpy as np
import mink
import mujoco
from ik_demo_utils import (
    waypoint_sequence, joint_space_lerp,
    compute_gripper_closing_axis_local, compute_plate_rim_grasp,
)

APPROACH_Z_OFFSET = 0.08
GRASP_OPEN = 1.2
GRASP_CLOSED = -0.16
CONTROL_DECIMATION = 10

# Dataset scope for this collection pass: open drawer -> close drawer -> pick
# plate. Cutlery retrieval is fully implemented below but SKIPPED, not
# deleted -- the moving-jaw convex hull cannot retain a thin cutlery capsule
# under lift, and a failed-grasp episode is worse training signal than no
# retrieve episode at all. Flip this back to False once that grasp is fixed
# (e.g. with the same connect-constraint trick used for the drawer handle);
# no other edit is needed.
SKIP_RETRIEVE = True

# Calibrated handle grasp pose (site position + orientation), measured
# from a manual viewer grasp where the fingers wrap the handle.
HANDLE_POS = np.array([-0.01374231, 0.12252117, 0.91137901])
HANDLE_QUAT = np.array([0.04128911, -0.13647269, 0.68135373, 0.71793281])

# Manually-verified left-arm "reach into the open drawer" pose (site sits at
# [0.02934, 0.05801, 0.82519]). Used as a joint-space seed so mink's Cartesian
# solver starts from -- and stays near -- the SAME elbow/wrist branch the
# person found by hand, instead of picking its own (which was hitting the
# front wall / housing lid / going to a completely different wrist_roll).
MANUAL_ENTRY_QPOS_5 = [-1.15, -0.14, 0.879, -0.0663, -1.46]  # 5 joints, no gripper
MANUAL_ENTRY_QUAT = np.array([0.62552, 0.68241, 0.10020, -0.36470])
LEFT_ARM_JOINTS_5 = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                     "left_wrist_flex", "left_wrist_roll"]

# Housing geometry (for the retreat-before-rising exit path):
# footprint y in [0.157, 0.323], lid bottom at world z ~= 0.900.
HOUSING_SAFE_Y_OUTSIDE = 0.15   # just in front of the open housing face
LID_CLEARANCE_Z = 0.895         # 5mm safety margin under the measured 0.900 lid

# Extra hold at the end of the episode: after the plate is picked up, keep
# the arm commanded to the same lifted pose for this many control steps of
# extra physics. Diagnostic only -- lets us watch in the viewer whether the
# weld/contact actually retains the plate or it slips. Set to 0 for dataset
# collection (real episodes should end shortly after the lift).
HOLD_STEPS_AFTER_LIFT = 300     # 300 * 10 substeps * 0.002s = 6.0s extra


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
    left_home_stage = np.array([-0.22, 0.00, 0.90])

    slide_axis = model.jnt_axis[model.joint("drawer_slide").id]
    slide_range = model.jnt_range[model.joint("drawer_slide").id][1]
    pull_target = HANDLE_POS + slide_axis * slide_range

    # -------- PHASE 1: right arm opens drawer --------
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
    grasp_activate_step = len(qpos_open) - 40

    scratch = _replay_phase_on_scratch(model, data.qpos.copy(), qpos_open, grip_open)

    # -------- PHASE 2: LEFT retrieves; RIGHT holds handle open --------
    qpos_retr = []
    grip_retr = []

    def add(segment_qpos, segment_grip):
        qpos_retr.extend(segment_qpos)
        grip_retr.extend(segment_grip)

    for name in ([] if SKIP_RETRIEVE else sorted(randomizer.last_in_drawer_items)):
        cutlery_pos = scratch.xpos[model.body(name).id].copy()

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", left_home_stage, None, GRASP_OPEN, None)],
            steps_per_segment=30,
            secondary_frame="right_gripperframe", secondary_xyz=pull_target,
            secondary_quat=HANDLE_QUAT, secondary_pos_cost=1.0, secondary_ori_cost=0.15,
        )
        add(seg, g)

        seg = joint_space_lerp(configuration, model, LEFT_ARM_JOINTS_5,
                                MANUAL_ENTRY_QPOS_5, steps=40)
        add(seg, [("left", GRASP_OPEN)] * len(seg))

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", cutlery_pos, MANUAL_ENTRY_QUAT, GRASP_OPEN, "drawer")],
            steps_per_segment=30,
        )
        add(seg, g)

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", cutlery_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        lift_pos = np.array([cutlery_pos[0], cutlery_pos[1], LID_CLEARANCE_Z])
        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", lift_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        retreat_pos = np.array([cutlery_pos[0], HOUSING_SAFE_Y_OUTSIDE, LID_CLEARANCE_Z])
        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", retreat_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        z_height = float(cutlery_pos[2])
        place_pos = np.array([-0.15, -0.15, z_height])
        seg, g = waypoint_sequence(
            configuration, model,
            [
                ("left_gripperframe", place_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, None),
                ("left_gripperframe", place_pos, None, GRASP_CLOSED, None),
                ("left_gripperframe", place_pos, None, GRASP_OPEN, None),
            ],
            steps_per_segment=20,
            secondary_frame="right_gripperframe", secondary_xyz=pull_target,
            secondary_quat=HANDLE_QUAT, secondary_pos_cost=1.0, secondary_ori_cost=0.15,
        )
        add(seg, g)

    # -------- PHASE 3: right closes drawer, left picks plate --------
    tail_waypoints = [
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_CLOSED),  # push closed
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_OPEN),    # release
        (handle_frame, right_home_stage, None, GRASP_OPEN),     # retract
    ]

    # Plate: grasp the RIM vertically, not the flat center. closing_axis_local
    # is a measured mechanism constant (independent of arm pose, verified
    # pose-invariant to ~1e-14); compute_plate_rim_grasp sweeps the remaining
    # roll DOF to find a kinematically comfortable rim point for THIS episode's
    # actual (randomized) plate position. It returns a candidate quat too, but
    # we deliberately pass target_quat=None on the plate waypoints: the 5-DOF
    # left arm cannot satisfy both a 6-DOF pose target and the rim position at
    # the same time, and a nonzero ori_cost makes it sacrifice position (the
    # arm ends up ~8cm above the plate, gripper closing in midair -- verified).
    # The physical finger-vs-rim contact does not determine success anyway:
    # the actual grasp is enforced by the left_grasp_weld equality constraint
    # anchored at first contact, so a rough wrist orientation is fine.
    plate_pos = data.xpos[model.body("plate").id].copy()
    left_base_xy = np.array([-0.22, 0.20])
    closing_axis_local = compute_gripper_closing_axis_local(
        model, data, "left_gripper", "left_moving_jaw_so101_v1", "left_gripper",
        reference_qpos_5=[0.0, 0.0, 0.0, 0.0, 0.0], arm_joint_names=LEFT_ARM_JOINTS_5,
    )
    rim_xyz, outside_xyz, plate_quat, plate_grasp_pos_err_cm, plate_grasp_angle_deg = compute_plate_rim_grasp(
        model, data, configuration,
        plate_center_xy=plate_pos[:2], left_base_xy=left_base_xy,
        plate_radius=0.089, plate_z=float(plate_pos[2]),
        closing_axis_local=closing_axis_local, arm_joint_names=LEFT_ARM_JOINTS_5,
        gripper_frame_name="left_gripperframe", home_stage_xyz=left_home_stage,
    )

    tail_waypoints.append(("left_gripperframe", left_home_stage, None, GRASP_OPEN, None))
    tail_waypoints.append(("left_gripperframe", outside_xyz, None, GRASP_OPEN, "plate"))
    tail_waypoints.append(("left_gripperframe", rim_xyz, None, GRASP_OPEN, "plate"))
    tail_waypoints.append(("left_gripperframe", rim_xyz, None, GRASP_CLOSED, "plate"))
    tail_waypoints.append(("left_gripperframe", rim_xyz + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, "plate"))

    # HOLD phase: keep the arm commanded to the same lifted pose for extra
    # physics steps after the lift completes. Diagnostic only -- lets us
    # watch in the viewer whether the weld/contact actually retains the plate
    # or it slips during the hold. Each repeat is a 20-step min-jerk segment
    # with start==target, so the arm holds still while mj_step keeps running.
    # HOLD_STEPS_AFTER_LIFT=300 gives 6 extra seconds; set to 0 to disable
    # for real dataset collection.
    if HOLD_STEPS_AFTER_LIFT > 0:
        lift_pose_xyz = rim_xyz + [0, 0, APPROACH_Z_OFFSET]
        n_hold_segments = HOLD_STEPS_AFTER_LIFT // 20
        for _ in range(n_hold_segments):
            tail_waypoints.append(("left_gripperframe", lift_pose_xyz, None, GRASP_CLOSED, "plate"))

    qpos_tail, grip_tail = waypoint_sequence(
        configuration, model, tail_waypoints, steps_per_segment=20
    )
    grasp_deactivate_step = len(qpos_open) + len(qpos_retr) + 20

    qpos_trace = qpos_open + qpos_retr + qpos_tail
    gripper_trace = grip_open + grip_retr + grip_tail
    grasp_events = {
        "activate_step": grasp_activate_step,
        "deactivate_step": grasp_deactivate_step,
        "eq_name": "right_grasp_connect",
        "body1": "right_moving_jaw_so101_v1",
        "body2": "drawer",
        "plate_grasp_pos_err_cm": plate_grasp_pos_err_cm,
        "plate_grasp_angle_deg": plate_grasp_angle_deg,
    }
    return qpos_trace, gripper_trace, grasp_events