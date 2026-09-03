# `/eval` — evaluation harness and Intel benchmark

**Branch:** `feature/eval-benchmark` · **Config:** `configs/eval.yaml` (+ `sim.yaml`, `randomization.yaml`, `inference.yaml`)

Two **separate entry points** that share this directory and little else.

## 1. Episode harness — deliverable #2 and #4

Drives the full pipeline: reset `/sim` at a given seed, step physics, query
`/inference` for actions, detect subtask completion, stop on success or timeout.
Runs across the 10 seeds in `configs/randomization.yaml` and writes one record per
episode — instruction, seed, subtask completion, outcome, timing — in the format
`configs/eval.yaml:logging` specifies.

That log is not incidental: it is the source of both the 10-seed success-rate
summary in the submission and the on-screen state for the demonstration video.

## 2. Intel benchmark — deliverable #3

Runs the OpenVINO pipeline **standalone, without stepping MuJoCo**, and reports
latency, throughput, device selection, and precision. Keeping physics out is the
whole point: it isolates model performance, which is what the 20-point OpenVINO
criterion is scored on.

It calls into `/inference` rather than re-implementing model loading, so the
benchmarked path is the path the harness actually runs.

## What does not go here

Scene construction (`/sim`), model internals (`/inference`), training (`/policy`).
The harness orchestrates; it does not reimplement.

## Definition of done

One command runs 10 seeds and prints a success rate; a second, independent
command produces the benchmark report — both on the Intel Core Ultra machine,
both reproducible from a clean clone.
