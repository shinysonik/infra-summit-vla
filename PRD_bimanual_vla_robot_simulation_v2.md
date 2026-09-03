# PRD: Bimanual VLA Robot Simulation — Intel Physical AI Online Challenge

**AI Infra Summit Hackathon — Team Project**
**Author:** Davide (repo owner / maintainer)
**Status:** Draft v2 — realigned to official Intel challenge brief

---

## Changelog from v1

v1 was written before we had the official challenge brief and assumed a generic
"web demo of a bimanual VLA robot" with cloud-hosted inference (Vercel + Render/
Railway + Hugging Face ZeroGPU + Supabase). That architecture **does not satisfy
the actual requirements**, for one core reason:

> The challenge requires the MuJoCo simulation *and* the AI/VLA/VLM inference
> pipeline to run and be benchmarked **on an Intel Core Ultra Series 2/3 system**,
> optimized with **OpenVINO**, for the final demonstration and submission.

A cloud GPU inference service (HF ZeroGPU) run from a remote backend cannot produce
the required Intel CPU/iGPU/NPU latency/throughput benchmark, and isn't what's being
judged (20 of 100 points are specifically "OpenVINO & Intel Core Ultra Optimization").
So the whole hosting/infra section, the "no ZeroGPU quota" risk, and the generic
robot/task open questions from v1 are replaced below. The web frontend idea isn't
gone — it's demoted to an optional local demo UI, not the deliverable.

---

## 1. Overview

**Task (per official brief):** Bimanual VLA Manipulation with Multi-Modal Reasoning
— Challenge Option: *Setting Up a Dinner Table*.

**Goal:** Build an end-to-end, reproducible pipeline where two simulated SO-101
arms in MuJoCo receive a natural-language instruction, perceive the scene through
simulated cameras, plan and execute a multi-step table-setting sequence (open
drawer, retrieve spoons/forks, pick up plate/cup, hand-off between arms, pour),
and run this inference **locally on an Intel Core Ultra Series 2/3 system**, with
the model optimized via OpenVINO (CPU/iGPU/NPU).

**Out of scope for v1 (unchanged):** real SO-101 hardware, sim-to-real transfer,
training a VLA from scratch (we fine-tune/distill an existing policy — SmolVLA,
Pi0.5, ACT, or another LeRobot-compatible imitation-learning policy).

**Confirmed (was "open question" in v1):**
- Robot: dual SO-101 arms (fixed by the challenge, not ALOHA or a custom rig).
- Simulator: MuJoCo (fixed by the challenge).
- Task: dinner-table setting, as specified in the brief (drawer → cutlery → plate
  → cup → pour, with at least one hand-off or complementary dual-arm action).
- Success metric: task completion across 10 randomized seeds (object placement,
  weight, friction, shape, lighting, background varied) — this *is* the rubric's
  "Robustness & Generalization" criterion (15 pts).

---

## 2. Architecture

Local-first pipeline, no cloud inference dependency for the graded deliverable.
A thin web viewer is optional/nice-to-have for live demo polish, not a
requirement.

```
┌───────────────────────────────────────────────────────────────────┐
│                  INTEL CORE ULTRA SERIES 2/3 SYSTEM                 │
│                                                                       │
│  ┌───────────────────┐      ┌───────────────────────────────────┐  │
│  │   MuJoCo Sim        │◄────►│   VLA Policy Runtime               │  │
│  │   - dual SO-101      │ obs  │   - OpenVINO IR (quantized)        │  │
│  │   - dinner-table      │      │   - device: CPU / iGPU / NPU       │  │
│  │     scene + assets    │─────►│   - takes: instruction + camera    │  │
│  │   - domain            │ act  │     frames + robot state           │  │
│  │     randomization      │      │   - returns: action vector         │  │
│  │     (10 seeds)         │      └───────────────────────────────────┘  │
│  └───────────────────┘                                               │
│            │                                                          │
│            ▼                                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │   Eval / Benchmark harness                                      │  │
│  │   - runs N episodes across seeds, logs success/failure          │  │
│  │   - Intel inference benchmark: latency, throughput, device,     │  │
│  │     precision                                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                     │
                     ▼ (offline, before the above — training hardware unconstrained)
┌───────────────────────────────────────────────────────────────────┐
│   TRAINING (any local GPU or cloud — not provided by Intel)          │
│   - Hugging Face LeRobot (or compatible) fine-tuning/distillation     │
│   - Candidate policies: SmolVLA, Pi0.5, ACT                           │
│   - Output: checkpoint → exported/converted to OpenVINO IR            │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│   OPTIONAL: local web viewer (three.js/urdf-loader) reading the       │
│   sim-server's local state over WS — for live demo polish only,       │
│   not required by the rubric. Runs on the same Intel machine or a     │
│   laptop on the same LAN; no cloud hosting needed.                    │
└───────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. (Offline, any hardware) Fine-tune/distill the chosen policy in MuJoCo/LeRobot
   on demonstration data for the dinner-table task.
2. Convert/quantize the trained model to OpenVINO IR.
3. On the Intel Core Ultra machine: MuJoCo steps the dual-arm scene; each step (or
   window of steps) the policy runtime is queried with the instruction, current
   camera observation, and robot state; it returns an action; MuJoCo applies it.
4. The eval harness repeats this across 10 randomized seeds, logs success/failure
   per seed, and produces the demonstration video + success-rate summary.
5. The benchmark script runs the same OpenVINO pipeline standalone and reports
   latency, throughput, device (CPU/iGPU/NPU), and precision.

---

## 3. Components

### 3.1 MuJoCo Simulation (`/sim`)
- **Stack:** MuJoCo (Python bindings), MJCF scene for dual SO-101 + dinner-table
  assets (drawer, spoons, forks, plate, cup, bottle, table).
- **Responsibilities:**
  - Scene definition: dual SO-101 rig, table, drawer, cutlery, plate, cup, bottle.
  - Domain randomization: object pose, weight, friction, shape variation, lighting,
    background — parametrized by seed, not hardcoded per-run.
  - Simulated camera(s) providing the observation stream the policy consumes.
  - Physics stepping (`mj_step`) driven by the eval harness.
- **No hardcoded values:** scene/asset paths, randomization ranges, and seed list
  come from a config file, not inlined in scripts.

### 3.2 VLA Policy & Training (`/policy`)
- **Stack:** Hugging Face LeRobot (or compatible tooling) for fine-tuning/
  distillation; candidate base policies: SmolVLA, Pi0.5, ACT.
- **Responsibilities:**
  - Data collection/formatting for the dinner-table task (teleoperation or
    scripted demonstrations in MuJoCo, per LeRobot dataset format).
  - Training/fine-tuning loop, checkpointing.
  - Export path to ONNX → OpenVINO IR conversion.
- **Config-driven:** model/checkpoint name, hyperparameters via config/env, not
  hardcoded in training scripts.

### 3.3 Inference Runtime (`/inference`)
- **Stack:** OpenVINO Runtime (Python API), targeting Intel CPU/iGPU/NPU.
- **Responsibilities:**
  - Load the OpenVINO IR model, select device at runtime (config-driven, not
    hardcoded — CPU vs iGPU vs NPU must be a parameter).
  - Expose a simple local interface (function call or lightweight local server)
    consumed by the MuJoCo eval loop: `(instruction, obs, robot_state) → action`.
  - Support batching/windowing of physics steps per inference call if latency
    requires it.
- **Modularity requirement:** the policy runtime is behind an interface so the
  underlying model (SmolVLA/Pi0.5/ACT/other) can be swapped without touching the
  sim loop or benchmark harness.

### 3.4 Eval & Benchmark Harness (`/eval`)
- **Responsibilities:**
  - Run the full pipeline across 10 randomized seeds, log per-seed
    success/failure and the sequence of sub-tasks completed (drawer open, item
    picked, hand-off performed, item placed, pour completed).
  - Produce the artifacts needed for the demonstration video (state capture per
    seed, or hooks to record).
  - Separate **Intel inference benchmark script**: runs the OpenVINO pipeline
    standalone (no need to run the full physics sim) and reports latency,
    throughput, device selection, and precision — this is deliverable #3 and
    directly maps to the 20-point OpenVINO/Intel Core Ultra rubric line.

### 3.5 Optional: Local Web Viewer (`/apps/web`, nice-to-have)
- **Stack:** Next.js/three.js/`urdf-loader`, run locally (no cloud hosting).
- **Purpose:** polish for the demonstration video only — visualize the sim state
  in a browser instead of (or alongside) MuJoCo's native viewer. Not part of the
  graded deliverables and not required to run the eval or benchmark.
- Cut entirely if time is tight — MuJoCo's built-in viewer/renderer is sufficient
  for the demo video requirement.

---

## 4. Hosting & Infra

No cloud hosting is required for the graded deliverable — everything must run
locally on the Intel Core Ultra machine for the final demo/benchmark anyway, so
provisioning cloud infra (Vercel/Render/Supabase/HF Spaces) would be extra
surface area that doesn't help the score.

| Component | Where it runs | Notes |
|---|---|---|
| MuJoCo sim + eval harness | Intel Core Ultra Series 2/3 machine | Required for final demo |
| OpenVINO inference | Same Intel machine (CPU/iGPU/NPU) | Required for benchmark deliverable |
| Training/fine-tuning | Any local GPU or cloud instance | Not provided by Intel; can be a teammate's GPU laptop or a rented cloud GPU for a few hours |
| Repo | GitHub | Required deliverable #1 |
| Optional web viewer | Local machine / LAN | Not required; cut if no time |

**Explicitly rejected (vs. v1):** hosting the inference pipeline on HF ZeroGPU or
any remote GPU service as the "production" path — it can't be what's benchmarked
or demonstrated on Intel hardware, so it would be wasted effort against the rubric.
It's still fine to use a cloud GPU purely for the *training* step, since training
hardware is explicitly unconstrained by the brief.

**Key logistics risk to resolve early:** at least one team member needs hands-on
(or remote) access to an actual Intel Core Ultra Series 2/3 machine for the final
run + benchmark + video. Confirm who has one, or whether the hackathon organizers
provide access, before committing to a build plan.

---

## 5. Non-Functional Requirements

- **No hardcoding, anywhere:** scene/asset paths, seeds/randomization ranges,
  model checkpoint names, inference device (CPU/iGPU/NPU), and thresholds — all
  via config files or environment variables. Any PR introducing a hardcoded value
  that should be config is rejected in review.
- **Modular, clean architecture:** clear boundaries between simulation, policy/
  training, inference runtime, and eval/benchmark harness. Each independently
  runnable and testable, per the "Technical Quality & Reproducibility" rubric
  line (10 pts).
- **Reproducibility:** deterministic setup instructions, pinned dependencies,
  one-command (or clearly documented multi-command) path from clone → run eval
  → run benchmark.
- **Observability:** log per-episode outcome (instruction, seed, sub-task
  completion, success/failure, timing) to support both the demo video and the
  10-seed success-rate summary required in the submission.

---

## 6. Repository & Workflow

- **Repo ownership:** Davide owns `main` and repo settings.
- **Branching model:** one feature branch per contributor/feature
  (`feature/<name>-<short-desc>`), no direct commits to `main`.
- **PRs:** all changes via pull request, reviewed and merged only by Davide.
- **Branch protection:** `main` protected, PR required, no force-push.
- **Suggested branch split (based on current team, updated for the real scope):**
  - `feature/mujoco-scene` — dual SO-101 + dinner-table MJCF, domain randomization
  - `feature/policy-training` — LeRobot fine-tuning/distillation (SmolVLA/Pi0.5/ACT)
  - `feature/openvino-inference` — model export/quantization to OpenVINO IR, runtime wrapper, device selection
  - `feature/eval-benchmark` — 10-seed eval harness + Intel benchmark script
  - `feature/demo-viewer` (optional) — local viewer / video capture tooling

---

## 7. Open Questions (to resolve with team before build week)

1. Which base policy do we fine-tune — SmolVLA, Pi0.5, or ACT? (Trade-off:
   SmolVLA/Pi0.5 are more "VLA-native" and score better on the reasoning
   criterion; ACT is simpler/faster to get working end-to-end first.)
2. Do we build demonstration data ourselves (teleop in MuJoCo) or is there an
   existing LeRobot dataset close enough to the dinner-table task to fine-tune
   from?
3. Who on the team has access to an Intel Core Ultra Series 2/3 machine (or NUC)
   for the final benchmark/demo run? This blocks deliverables #3 and #4.
4. How do we split "reasoning" between the VLA policy itself and an auxiliary
   LLM/VLM layer (e.g. a separate instruction-parsing/planning step feeding the
   low-level policy)? This affects both the architecture and how much of the 20
   reasoning points we can credibly claim.
5. What's our target OpenVINO precision (FP16 vs INT8) and device (CPU/iGPU/NPU)
   — do we benchmark all three, or pick one and justify it?
