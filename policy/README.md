# `/policy` — VLA policy training and export

**Branch:** `feature/policy-training` · **Config:** `configs/policy.yaml`

Fine-tuning or distillation of the base policy, and the export path that hands a
model to `/inference`. This is the only module allowed to need a GPU: the brief
leaves training hardware unconstrained, and nothing here runs on the Intel demo
machine.

## What goes here

- **Demonstration data** — collection (scripted or teleop rollouts in `/sim`) and
  formatting into the LeRobot dataset layout.
- **Training loop** — LeRobot (or compatible) fine-tuning with checkpointing.
  Model name, hyperparameters, and dataset id all come from `configs/policy.yaml`.
- **Export** — checkpoint → ONNX → OpenVINO IR, written to the directory that
  `configs/inference.yaml` reads. Keep the two configs in sync.

## Open team decisions — do not silently resolve

1. **Base policy** (PRD §7.1): SmolVLA / Pi0.5 / ACT. SmolVLA and Pi0.5 are more
   VLA-native and score better against the 20-point reasoning criterion; ACT is
   quicker to get working end to end. `base_policy` is `null` in the config on
   purpose.
2. **Data source** (PRD §7.2): self-collected demonstrations vs. an existing
   LeRobot dataset close enough to fine-tune from.
3. **Reasoning split** (PRD §7.4): how much sits in the VLA policy itself vs. an
   auxiliary LLM/VLM planning layer. This changes both the architecture and how
   much of the reasoning score is credible.

## Definition of done

A checkpoint that produces sane actions in `/sim`, and an OpenVINO IR export that
`/inference` can load — reproducible from a documented command.
