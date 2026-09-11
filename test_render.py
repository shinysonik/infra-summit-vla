import os
import mujoco
import time

# Загрузи модель
model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)

# Создай renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Прогрей
for _ in range(10):
    renderer.update_scene(data, camera="overhead")
    renderer.render()

# Замерь
start = time.time()
N = 100
for _ in range(N):
    renderer.update_scene(data, camera="overhead")
    renderer.render()
elapsed = time.time() - start
fps = N / elapsed

print(f"Render FPS: {fps:.1f}")
print(f"Time per frame: {1000/fps:.1f} ms")

# Проверь, какой OpenGL-драйвер активен
import ctypes
try:
    opengl32 = ctypes.windll.opengl32
    vendor = ctypes.create_string_buffer(256)
    # Простой способ — посмотреть через переменные окружения
except Exception as e:
    pass

print("\nMUJOCO_GL =", os.environ.get("MUJOCO_GL", "not set"))