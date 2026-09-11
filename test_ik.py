import mink, mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
config = mink.Configuration(model)

targets = [
    np.array([0.0, -0.1, 0.85]),
    np.array([0.15, 0.0, 0.85]),
    np.array([-0.15, 0.0, 0.85]),
    np.array([0.0, 0.2, 0.85]),
    np.array([0.0, -0.25, 0.85]),
]

for t in targets:
    # Сброс конфигурации перед каждой целью
    config.update(config.model.qpos0)
    
    task = mink.FrameTask(
        frame_name="left_gripper",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    task.set_target(mink.SE3.from_translation(t))
    
    for _ in range(300):
        vel = mink.solve_ik(config, [task], dt=0.01, solver="daqp")
        config.integrate_inplace(vel, dt=0.01)
    
    # Правильно: получаем реальную позицию gripper после IK
    mujoco.mj_forward(model, config.data)
    gripper_pos = config.data.xpos[model.body("right_gripper").id]
    error = np.linalg.norm(gripper_pos - t)
    print(f"Target {t}: error = {error*100:.2f} cm")