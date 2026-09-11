import mink, mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
config = mink.Configuration(model)
lg_id = model.body("left_gripper").id
rg_id = model.body("right_gripper").id

# Сетка точек на высоте стола (z=0.85)
xs = np.linspace(-0.40, 0.40, 9)
ys = np.linspace(-0.10, 0.35, 5)

print("X (см) | Y (см) | Left err | Right err")
print("-" * 50)

for y in ys:
    for x in xs:
        target = np.array([x, y, 0.85])
        
        # Левая
        config.update(config.model.qpos0)
        task_l = mink.FrameTask("left_gripper", "body", position_cost=1.0, orientation_cost=0.0)
        task_l.set_target(mink.SE3.from_translation(target))
        for _ in range(300):
            vel = mink.solve_ik(config, [task_l], dt=0.01, solver="daqp")
            config.integrate_inplace(vel, dt=0.01)
        mujoco.mj_forward(model, config.data)
        err_l = np.linalg.norm(config.data.xpos[lg_id] - target) * 100
        
        # Правая
        config.update(config.model.qpos0)
        task_r = mink.FrameTask("right_gripper", "body", position_cost=1.0, orientation_cost=0.0)
        task_r.set_target(mink.SE3.from_translation(target))
        for _ in range(300):
            vel = mink.solve_ik(config, [task_r], dt=0.01, solver="daqp")
            config.integrate_inplace(vel, dt=0.01)
        mujoco.mj_forward(model, config.data)
        err_r = np.linalg.norm(config.data.xpos[rg_id] - target) * 100
        
        ok = "✅" if (err_l < 3 or err_r < 3) else "❌"
        both = "🎯" if (err_l < 3 and err_r < 3) else "  "
        print(f"{x*100:+6.1f} | {y*100:+6.1f} | {err_l:6.2f}   | {err_r:6.2f}   {ok} {both}")