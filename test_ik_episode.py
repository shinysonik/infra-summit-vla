import mujoco
import sys
sys.path.insert(0, "sim")

from randomization import DomainRandomizer
from scripted_episode_ik import build_drawer_and_cutlery_episode

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)
randomizer = DomainRandomizer(model, "configs/randomization.yaml")

randomizer.reset(data, seed=10001)
mujoco.mj_forward(model, data)

print("Testing episode build...")
try:
    qpos_trace, gripper_trace = build_drawer_and_cutlery_episode(model, data, randomizer)
    print(f"OK: episode built, {len(qpos_trace)} steps")
    print(f"Cutlery in drawer: {randomizer.last_in_drawer_items}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()