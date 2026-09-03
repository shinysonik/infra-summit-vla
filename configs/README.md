# `configs/` — single source of truth for everything configurable

The PRD's hardest rule (§5) is **no hardcoded values, anywhere**: scene and asset
paths, seed lists, randomization ranges, model checkpoint names, inference device
(CPU/iGPU/NPU), precision, and thresholds all live here, never as constants in
code. A PR that inlines a value that belongs in this directory gets rejected in
review.

Keeping the configs in one directory — rather than one per module — is deliberate:
it gives review a single place to check that rule, and it lets `/sim` and `/eval`
share `randomization.yaml` (the 10-seed list must have exactly one definition in
the repo) without duplicating it.

| File | Owned by | Read by |
|---|---|---|
| `sim.yaml` | `feature/mujoco-scene` | `/sim`, `/eval` |
| `randomization.yaml` | `feature/mujoco-scene` | `/sim`, `/eval` |
| `policy.yaml` | `feature/policy-training` | `/policy` |
| `inference.yaml` | `feature/openvino-inference` | `/inference`, `/eval` |
| `eval.yaml` | `feature/eval-benchmark` | `/eval` |

## Conventions

- **Paths are relative to the repository root**, so a config works from any
  working directory once the loader resolves against the repo root.
- **`null` means "undecided", not "default"**. Several values are open team
  questions (base policy, dataset source, target precision/device — PRD §7).
  An implementation session must surface those, not quietly pick one.
- **Environment variables may override any key** once a loader exists, for the
  values that differ per machine (notably `device` on the Intel box). The file
  stays the documented default; the env var is the escape hatch.
- **Cross-file contracts:** `policy.yaml:export.openvino_ir_dir` must stay in
  sync with `inference.yaml:model.ir_xml`/`ir_bin`. If you move one, move both.
