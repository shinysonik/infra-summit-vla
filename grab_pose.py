"""Interactive pose reader. Drag the right arm in the viewer until the
gripper wraps the handle, then read the printed qpos values from the terminal.
Ctrl+drag to move, right mouse to rotate, scroll to zoom. Close window to exit.
"""
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)

# Same right-arm joints the IK script writes to.
JOINTS = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Drag the right arm to the handle grasp pose.")
    print("Press Ctrl+C here to print the current qpos.\n")
    step = 0
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            if step % 300 == 0:  # every ~1s at 200Hz
                qpos = []
                for jn in JOINTS:
                    jid = model.joint(jn).id
                    adr = model.jnt_qposadr[jid]
                    qpos.append(data.qpos[adr])
                print("qpos:", np.array2string(np.array(qpos), precision=4))
    except KeyboardInterrupt:
        pass