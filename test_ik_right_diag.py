import mink, mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
config = mink.Configuration(model)

# 1. Проверь имена тел — существуют ли они?
print("Bodies found:")
for name in ["left_gripper", "right_gripper", "left_base", "right_base"]:
    try:
        bid = model.body(name).id
        print(f"  {name}: OK, id={bid}")
    except Exception as e:
        print(f"  {name}: FAILED — {e}")

# 2. Где обе руки находятся в нейтральной позе (qpos0)?
config.update(config.model.qpos0)
mujoco.mj_forward(model, config.data)
lg_id = model.body("left_gripper").id
rg_id = model.body("right_gripper").id
print(f"\nAt qpos0:")
print(f"  left_gripper:  {config.data.xpos[lg_id]}")
print(f"  right_gripper: {config.data.xpos[rg_id]}")

# 3. Попробуй достать до цели ПРЯМО ПЕРЕД правой базой
target = np.array([0.16, 0.10, 0.85])
config.update(config.model.qpos0)
task = mink.FrameTask("right_gripper", "body", position_cost=1.0, orientation_cost=0.0)
task.set_target(mink.SE3.from_translation(target))
for i in range(1000):
    vel = mink.solve_ik(config, [task], dt=0.01, solver="daqp")
    config.integrate_inplace(vel, dt=0.01)
mujoco.mj_forward(model, config.data)
final_pos = config.data.xpos[rg_id]
error = np.linalg.norm(final_pos - target)
print(f"\nTrying to reach {target} with right_gripper")
print(f"  Final position: {final_pos}")
print(f"  Error: {error*100:.2f} cm")
print(f"  Full qpos: {config.q}")
print(f"  Right shoulder_pan: {config.q[6]:.3f} rad")

# 4. То же самое для левой руки (контроль)
target_l = np.array([-0.16, 0.10, 0.85])
config.update(config.model.qpos0)
task_l = mink.FrameTask("left_gripper", "body", position_cost=1.0, orientation_cost=0.0)
task_l.set_target(mink.SE3.from_translation(target_l))
for i in range(1000):
    vel = mink.solve_ik(config, [task_l], dt=0.01, solver="daqp")
    config.integrate_inplace(vel, dt=0.01)
mujoco.mj_forward(model, config.data)
final_pos_l = config.data.xpos[lg_id]
error_l = np.linalg.norm(final_pos_l - target_l)
print(f"\nTrying to reach {target_l} with left_gripper (control)")
print(f"  Error: {error_l*100:.2f} cm")