# `/sim` — MuJoCo simulation

**Branch:** `feature/mujoco-scene` · **Config:** `configs/sim.yaml`, `configs/randomization.yaml`

The dual SO-101 dinner-table scene and everything that varies it. This module owns
the world; it does not own the control loop — `/eval` drives `mj_step`.

## What goes here

- **`assets/`** — the MJCF scene and its meshes/textures: two SO-101 arms on a
  table, plus drawer, spoons, forks, plate, cup, bottle. Robot and simulator are
  fixed by the challenge; they are not choices to revisit.
- **Scene loading** — resolve the MJCF and asset paths from `configs/sim.yaml`,
  build the `mjModel`/`mjData`, expose the named bodies/joints the task needs.
- **Domain randomization** — a `seed -> randomized scene` function driven entirely
  by `configs/randomization.yaml`: object pose, mass, friction, shape, lighting,
  background. Applied at episode reset, never mid-episode. This is what the
  15-point *Robustness & Generalization* criterion measures.
- **Observation assembly** — render the cameras declared in `configs/sim.yaml`
  and gather robot state into the observation the policy consumes.

## What does not go here

Policy inference, OpenVINO, success criteria, and episode orchestration. `/sim`
exposes reset/step/observe and stays ignorant of what is choosing the actions —
that boundary is what lets `/inference` swap the underlying model without
touching the scene.

## Definition of done

A seeded reset produces a visibly different but still solvable scene, cameras
render, and the arms are controllable — with no path, seed, or range written in
Python.
