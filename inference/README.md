# `/inference` — OpenVINO runtime

**Branch:** `feature/openvino-inference` · **Config:** `configs/inference.yaml`

The only module that talks to OpenVINO. Worth 20 of 100 rubric points on its own,
and the reason the whole project is local-first: a cloud GPU cannot produce the
Intel Core Ultra benchmark the brief requires.

## What goes here

- **IR loading and device selection** — compile the model for the configured
  device. `CPU`, `GPU` (Intel iGPU), `NPU`, or `AUTO` must be a config value;
  switching targets is a config edit, never a code edit.
- **The policy interface** — one call:

      (instruction, observation, robot_state) -> action

  `/eval` and `/sim` know only this signature. The underlying model
  (SmolVLA / Pi0.5 / ACT / whatever replaces it) stays swappable behind it —
  that modularity is an explicit PRD requirement, not a nicety.
- **Preprocessing** — image and state tensors shaped for the IR, kept next to the
  runtime so the sim loop never learns the model's input layout.
- **Action chunking** — return a window of actions per call when per-step latency
  is too high for real-time stepping (`action_chunk_size`).

## What does not go here

Timing and reporting. The benchmark lives in `/eval` and calls into this module,
so the numbers it reports come from the same code path the eval harness uses.

## Open team decision

**Precision and device** (PRD §7.5): FP16 vs INT8, and whether to benchmark
CPU/iGPU/NPU all three or pick one and justify it. The rubric explicitly weighs
quantization choice and device utilization *and* requires that optimization not
degrade task success — so any precision drop needs a success-rate number beside
it, not just a latency number.

## Definition of done

The same call works on CPU, iGPU, and NPU by changing one config key, and the
task success rate is measured on each configuration that gets reported.
