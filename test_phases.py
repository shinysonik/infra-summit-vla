import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
sys.path.insert(0, "sim")
sys.path.insert(0, ".")

from randomization import DomainRandomizer
from scripted_episode_ik import build_drawer_and_cutlery_episode
from ik_demo_utils import activate_grasp_connect, deactivate_grasp_connect

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)
randomizer = DomainRandomizer(model, "configs/randomization.yaml")
randomizer.reset(data, 10001)
mujoco.mj_forward(model, data)

bid = model.body("right_gripper").id
OBJECTS = ["plate", "cup", "bottle",
           "spoon_1", "spoon_2", "fork_1", "fork_2"]

ACTUATORS = ["left_shoulder_pan", "left_shoulder_lift",
             "left_elbow_flex", "left_wrist_flex",
             "left_wrist_roll", "left_gripper",
             "right_shoulder_pan", "right_shoulder_lift",
             "right_elbow_flex", "right_wrist_flex",
             "right_wrist_roll", "right_gripper"]

start_xy = {n: data.xpos[model.body(n).id][:2].copy() for n in OBJECTS}
drawer_jid = model.joint("drawer_slide").id
drawer_qadr = model.jnt_qposadr[drawer_jid]
drawer_start = data.qpos[drawer_qadr]

print(f"Drawer start position: {drawer_start:.4f}\n")

qpos_trace, gripper_trace, grasp_events = build_drawer_and_cutlery_episode(model, data, randomizer)
print(f"Trajectory built: {len(qpos_trace)} steps\n")

# Replay with live viewer
viewer = mujoco.viewer.launch_passive(model, data)



first_move = {}
for step in range(len(qpos_trace)):
    if not viewer.is_running():
        break
    for act_name in ACTUATORS:
        if act_name.endswith("_gripper"):
            continue  # gripper ctrl comes from gripper_trace below only
        act_id = model.actuator(act_name).id
        jid = model.actuator_trnid[act_id, 0]
        adr = model.jnt_qposadr[jid]
        data.ctrl[act_id] = qpos_trace[step][adr]

    gval = gripper_trace[step]
    if gval is not None:
        side, val = gval
        data.ctrl[model.actuator(f"{side}_gripper").id] = val

    if step == grasp_events["activate_step"]:
        handle_world_now = data.geom_xpos[model.geom("drawer_handle").id].copy()
        activate_grasp_connect(model, data, grasp_events["eq_name"],
                                grasp_events["body1"], grasp_events["body2"],
                                handle_world_now)
    if step == grasp_events["deactivate_step"]:
        deactivate_grasp_connect(model, data, grasp_events["eq_name"])

    for _ in range(10):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.005)

    # Full-span left-arm contact check -- every step from the start of
    # retrieve to the start of tail, not just one snapshot. This is the
    # part I cannot verify without your real meshes (my sandbox only has
    # sphere-proxy collision geometry); this loop gives you a definitive
    # pass/fail the moment you run it.
    if grasp_events["activate_step"] <= step <= grasp_events["deactivate_step"]:
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            if ("left_" in n1 or "left_" in n2) and con.dist < -0.0005:
                other = n2 if "left_" in n1 else n1
                if "left_" in other:
                    continue  # left-arm self contact, not an obstacle hit
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                print(f"    [step {step}] LEFT ARM HIT: {n1}({g1}) <-> {n2}({g2})  pen={con.dist:.4f}")

    if step in (0, 100, 199, 200, 220, 240, 260, 280, 300, 320, 340, 400, 499):
        L_site = data.site_xpos[model.site("left_gripperframe").id]
        R_site = data.site_xpos[model.site("right_gripperframe").id]
        print(f"step {step:4d}  drawer={data.qpos[drawer_qadr]:.3f}  "
              f"R_site=[{R_site[0]:.3f} {R_site[1]:.3f} {R_site[2]:.3f}]  "
              f"L_site=[{L_site[0]:.3f} {L_site[1]:.3f} {L_site[2]:.3f}]")
        if step == 240:
            print("  --- step 240: IK-plan vs physics ---")
            for jn in ["left_shoulder_pan", "left_shoulder_lift",
                       "left_elbow_flex", "left_wrist_flex", "left_wrist_roll"]:
                jid = model.joint(jn).id
                adr = model.jnt_qposadr[jid]
                planned = qpos_trace[step][adr]
                actual = data.qpos[adr]
                print(f"    {jn}: planned={planned:+.4f} actual={actual:+.4f} diff={actual-planned:+.4f}")
    for n in OBJECTS:
        if n in first_move:
            continue
        shift = float(np.linalg.norm(data.xpos[model.body(n).id][:2] - start_xy[n]))
        if shift > 0.03:
            first_move[n] = step

viewer.close()

print("=== STEP 3: first move of each object >3cm ===")
if not first_move:
    print("  no objects moved")
else:
    for n, step in first_move.items():
        print(f"  {n} at step {step}")

drawer_end = data.qpos[drawer_qadr]
print(f"\n=== Drawer motion ===")
print(f"  start: {drawer_start:.4f}")
print(f"  end:   {drawer_end:.4f}")
print(f"  travel: {(drawer_end - drawer_start)*100:.2f} cm")

print(f"\n=== Final object shifts ===")
for n in OBJECTS:
    shift = float(np.linalg.norm(data.xpos[model.body(n).id][:2] - start_xy[n]))
    print(f"  {n}: {shift*100:.1f} cm")

print("\n=== cutlery position after replay (drawer open, real) ===")
for n in sorted(randomizer.last_in_drawer_items):
    print(f"  {n}: {data.xpos[model.body(n).id][:2]}")