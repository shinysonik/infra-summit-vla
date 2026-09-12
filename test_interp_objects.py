import mujoco
import numpy as np
import sys
sys.path.insert(0, "sim")
sys.path.insert(0, ".")

from randomization import DomainRandomizer
from scripted_episode_ik import build_drawer_and_cutlery_episode

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)
randomizer = DomainRandomizer(model, "configs/randomization.yaml")
randomizer.reset(data, 10001)
mujoco.mj_forward(model, data)

qpos_trace, _ = build_drawer_and_cutlery_episode(model, data, randomizer)

print("qpos_trace[0]  obj part:", qpos_trace[0][0:7])
print("qpos_trace[-1] obj part:", qpos_trace[-1][0:7])

if np.allclose(qpos_trace[0][0:7], qpos_trace[-1][0:7]):
    print("Objects NOT touched by interp.")
else:
    print("Objects ARE being dragged by interp -- this is the bug.")
