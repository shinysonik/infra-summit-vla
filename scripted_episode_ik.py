"""
Replaces the sine-sweep placeholder in collect_demonstrations.py with real
IK-generated trajectories, automatically built from wherever DomainRandomizer
actually placed objects/drawer THIS seed -- no manual per-seed tuning.

Encodes the behavior discussed: close the drawer after every retrieval,
never place/move dishes while the drawer is still open.
"""
import numpy as np
import mink
import mujoco
from ik_demo_utils import waypoint_sequence

APPROACH_Z_OFFSET = 0.08   # hover height above a grasp target before descending
GRASP_OPEN = 1.2           # gripper joint value: open
GRASP_CLOSED = 0.0         # gripper joint value: closed


def build_drawer_and_cutlery_episode(model, data, randomizer):
    """Phase order enforces your stated policy directly: open -> retrieve ->
    CLOSE -> only then touch any dish. The drawer is never left open while
    a dish waypoint runs."""
    configuration = mink.Configuration(model)
    configuration.update(data.qpos.copy())

    handle_pos = data.geom_xpos[model.geom("drawer_handle").id].copy()
    handle_pos = handle_pos + np.array([0, -0.03, 0])
    drawer_bid = model.body("drawer").id
    # pick whichever gripper the reach map says is reliable for the handle
    # (right, given the current between-arms layout at x=0) -- don't hardcode
    # this if you later move the drawer; read it from your reach map instead.
    handle_frame = "right_gripper"

    waypoints = []
    # 1. hover above handle
    waypoints.append((handle_frame, handle_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN))
    # 2. descend to handle, gripper open
    waypoints.append((handle_frame, handle_pos, None, GRASP_OPEN))
    # 3. close gripper around handle
    waypoints.append((handle_frame, handle_pos, None, GRASP_CLOSED))
    # 4. pull open -- move gripper along the drawer's own slide axis by its
    #    real travel range (read from the model, not hardcoded)
    slide_axis = model.jnt_axis[model.joint("drawer_slide").id]
    slide_range = model.jnt_range[model.joint("drawer_slide").id][1]
    pull_target = handle_pos + slide_axis * slide_range
    waypoints.append((handle_frame, pull_target, None, GRASP_CLOSED))
    # 5. release handle
    waypoints.append((handle_frame, pull_target, None, GRASP_OPEN))

    # 6. retrieve cutlery THIS seed actually put in the drawer -- read real
    #    positions from randomizer state, not assumed coordinates
    for name in randomizer.last_in_drawer_items:
        cutlery_pos = data.xpos[model.body(name).id].copy()
        frame = handle_frame  # or pick nearer arm by x-distance if you want both arms working
        waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN))
        waypoints.append((frame, cutlery_pos, None, GRASP_OPEN))
        waypoints.append((frame, cutlery_pos, None, GRASP_CLOSED))
        waypoints.append((frame, cutlery_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED))
        # place it somewhere reachable on the table -- use your reach map's
        # 🎯 zone, not an arbitrary point
        place_pos = np.array([0.0, 0.10, cutlery_pos[2]])
        waypoints.append((frame, place_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED))
        waypoints.append((frame, place_pos, None, GRASP_CLOSED))
        waypoints.append((frame, place_pos, None, GRASP_OPEN))

    # 7. CLOSE the drawer -- mandatory, every episode, before any dish moves
    waypoints.append((handle_frame, pull_target, None, GRASP_OPEN))
    waypoints.append((handle_frame, pull_target, None, GRASP_CLOSED))
    close_target = handle_pos
    waypoints.append((handle_frame, close_target, None, GRASP_CLOSED))
    waypoints.append((handle_frame, close_target, None, GRASP_OPEN))

    # 8. only now touch dishes -- pick arm by which SIDE the object is on,
    #    matching your reach map (left covers negative x, right covers
    #    positive x, near x=0 either works). Hardcoding one arm here was
    #    the bug the waypoint_sequence warning just caught (81.54cm error).
    plate_pos = data.xpos[model.body("plate").id].copy()
    plate_arm = "left_gripper" if plate_pos[0] < 0 else "right_gripper"
    waypoints.append((plate_arm, plate_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_OPEN))
    # ... continue the same pick/place pattern for bowl, cup, bottle

    qpos_trace, gripper_trace = waypoint_sequence(configuration, model, waypoints)
    return qpos_trace, gripper_trace
