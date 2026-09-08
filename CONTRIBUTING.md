# Contributing / Участие в проекте

> **EN** — Contribution workflow for this repository. Russian version below.
> **RU** — Правила работы с репозиторием. Английская версия выше.

---

## 🇬🇧 English

### 1. Push policy

`main` is unprotected: everyone pushes freely, direct commits included. No
required Pull Request, no branch restrictions.

### 2. Branch naming (convention, not enforced)

One branch per contributor/feature is still the suggested convention:

```
feature/<short-description>
```

Planned split for this project:

| Branch | Scope |
|---|---|
| `feature/mujoco-scene` | Dual SO-101 + dinner-table MJCF, domain randomization |
| `feature/policy-training` | LeRobot fine-tuning / distillation |
| `feature/openvino-inference` | ONNX → OpenVINO IR, runtime wrapper, device selection |
| `feature/eval-benchmark` | 10-seed eval harness + Intel benchmark script |
| `feature/demo-viewer` | Optional local viewer / capture tooling |

Direct commits to `main` are fine too — use your judgement on when a feature
branch is worth it.

### 3. Code rules that get PRs rejected (when you do open one)

- **No hardcoded values.** Scene/asset paths, seeds, randomization ranges, model
  checkpoint names, inference device (CPU/iGPU/NPU), precision, thresholds — all
  live in `configs/*.yaml`. This is PRD §5 and it is enforced in review.
- **Respect module boundaries.** `/sim`, `/policy`, `/inference`, `/eval` must
  each stay independently runnable and testable. Do not reach across them.
- **Log per episode.** Instruction, seed, subtask completion, outcome, timing —
  the submission's success-rate summary is built from those logs.

### 4. Do not decide these alone

The base policy, the demonstration-data source, the reasoning split, and the
target precision/device are open team questions (PRD §7). Raise them; don't
quietly pick one in a PR.

---

## 🇷🇺 Русский

### 1. Политика пушей

Ветка `main` не защищена: все пушат свободно, включая прямые коммиты. PR не
обязателен, ограничений на ветки нет.

### 2. Именование веток (рекомендация, не требование)

Одна ветка на участника/задачу по-прежнему рекомендуется:

```
feature/<краткое-описание>
```

Запланированное разделение:

| Ветка | Содержание |
|---|---|
| `feature/mujoco-scene` | MJCF-сцена: два SO-101 + обеденный стол, рандомизация |
| `feature/policy-training` | Дообучение / дистилляция политики (LeRobot) |
| `feature/openvino-inference` | ONNX → OpenVINO IR, обёртка рантайма, выбор устройства |
| `feature/eval-benchmark` | Оценка на 10 сидах + бенчмарк для Intel |
| `feature/demo-viewer` | Опциональный локальный просмотрщик |

Прямые коммиты в `main` тоже допустимы — решайте сами, когда нужна отдельная ветка.

### 3. Из-за чего PR отклоняют (если всё же открываете PR)

- **Никаких захардкоженных значений.** Пути к сцене и ассетам, сиды, диапазоны
  рандомизации, имена чекпоинтов, устройство инференса (CPU/iGPU/NPU), точность,
  пороги — всё это в `configs/*.yaml`. Это требование PRD §5, и оно проверяется
  на ревью.
- **Соблюдайте границы модулей.** `/sim`, `/policy`, `/inference`, `/eval`
  должны запускаться и тестироваться независимо друг от друга.
- **Логируйте каждый эпизод.** Инструкция, сид, выполненные подзадачи, результат,
  тайминги — итоговая статистика успеха собирается из этих логов.

### 4. Что нельзя решать в одиночку

Базовая политика, источник демонстрационных данных, распределение «рассуждения»
между политикой и отдельным LLM/VLM-слоем, целевая точность и устройство —
открытые командные вопросы (PRD §7). Выносите их на обсуждение, а не решайте
молча внутри PR.
