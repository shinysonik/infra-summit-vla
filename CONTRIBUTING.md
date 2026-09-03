# Contributing / Участие в проекте

> **EN** — Contribution workflow for this repository. Russian version below.
> **RU** — Правила работы с репозиторием. Английская версия выше.

---

## 🇬🇧 English

### 1. First-time setup

After cloning, install the shared git hooks **once**:

```bash
git clone https://github.com/Akaired/infra-summit-vla.git
cd infra-summit-vla
./scripts/setup-hooks.sh
```

That points git at the versioned `.githooks/` directory
(`git config core.hooksPath .githooks`). Without it you lose the local warning
described in §4 — the GitHub-side protection still applies either way.

### 2. Branch naming

One branch per contributor/feature:

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

### 3. Workflow

```bash
git checkout main
git pull
git checkout -b feature/my-feature
# ... commit your work ...
git push -u origin feature/my-feature
```

Then open a Pull Request on GitHub. **Davide (@Akaired) reviews and merges.**
Nobody merges their own PR into `main`.

### 4. `main` is protected — two layers

**Layer 1 — GitHub branch protection (the real enforcement).**
`main` requires a Pull Request, rejects force-pushes, and rejects branch
deletion. A direct push is refused by the server with GitHub's standard English
message, roughly:

```
remote: error: GH006: Protected branch update failed for refs/heads/main.
```

That message **cannot be customised**: github.com does not support custom
server-side `pre-receive` hooks — those exist only on GitHub Enterprise Server.

**Layer 2 — the local `pre-push` hook (a reminder, not enforcement).**
Because the server message cannot be translated, `.githooks/pre-push` catches the
push *before* it leaves your machine and explains the policy in English and
Russian. It exits non-zero and the push never happens.

Understand the difference: the hook is a courtesy that makes the failure
readable. **It is not the security boundary.** Anyone can skip it with
`git push --no-verify`, or simply never run `setup-hooks.sh` — and `main` is
still protected, because layer 1 lives on GitHub and does not depend on anything
in your working copy.

### 5. Code rules that get PRs rejected

- **No hardcoded values.** Scene/asset paths, seeds, randomization ranges, model
  checkpoint names, inference device (CPU/iGPU/NPU), precision, thresholds — all
  live in `configs/*.yaml`. This is PRD §5 and it is enforced in review.
- **Respect module boundaries.** `/sim`, `/policy`, `/inference`, `/eval` must
  each stay independently runnable and testable. Do not reach across them.
- **Log per episode.** Instruction, seed, subtask completion, outcome, timing —
  the submission's success-rate summary is built from those logs.

### 6. Do not decide these alone

The base policy, the demonstration-data source, the reasoning split, and the
target precision/device are open team questions (PRD §7). Raise them; don't
quietly pick one in a PR.

---

## 🇷🇺 Русский

### 1. Первоначальная настройка

После клонирования **один раз** установите общие git-хуки:

```bash
git clone https://github.com/Akaired/infra-summit-vla.git
cd infra-summit-vla
./scripts/setup-hooks.sh
```

Скрипт указывает git на каталог `.githooks/`, который хранится в репозитории
(`git config core.hooksPath .githooks`). Без этого вы не увидите локальное
предупреждение из §4 — защита на стороне GitHub при этом действует в любом случае.

### 2. Именование веток

Одна ветка на участника/задачу:

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

### 3. Рабочий процесс

```bash
git checkout main
git pull
git checkout -b feature/моя-задача
# ... коммиты ...
git push -u origin feature/моя-задача
```

Затем откройте Pull Request на GitHub. **Проверяет и мержит Davide (@Akaired).**
Никто не мержит собственный PR в `main`.

### 4. Ветка `main` защищена — два уровня

**Уровень 1 — branch protection на GitHub (настоящая защита).**
Для `main` обязателен Pull Request, force-push и удаление ветки запрещены.
Прямой пуш отклоняется сервером стандартным сообщением GitHub на английском:

```
remote: error: GH006: Protected branch update failed for refs/heads/main.
```

Это сообщение **нельзя изменить**: github.com не поддерживает пользовательские
серверные хуки `pre-receive` — они есть только в GitHub Enterprise Server.

**Уровень 2 — локальный хук `pre-push` (напоминание, а не защита).**
Поскольку серверное сообщение нельзя перевести, `.githooks/pre-push`
перехватывает пуш ещё *до* отправки с вашей машины и объясняет правила
по-английски и по-русски. Он завершается с ненулевым кодом, и пуш не выполняется.

Важно понимать разницу: хук — это удобство, которое делает ошибку понятной.
**Он не является границей безопасности.** Его можно обойти командой
`git push --no-verify` или просто не запускать `setup-hooks.sh` — и `main`
всё равно останется защищённой, потому что уровень 1 живёт на GitHub и не
зависит от вашей рабочей копии.

### 5. Из-за чего PR отклоняют

- **Никаких захардкоженных значений.** Пути к сцене и ассетам, сиды, диапазоны
  рандомизации, имена чекпоинтов, устройство инференса (CPU/iGPU/NPU), точность,
  пороги — всё это в `configs/*.yaml`. Это требование PRD §5, и оно проверяется
  на ревью.
- **Соблюдайте границы модулей.** `/sim`, `/policy`, `/inference`, `/eval`
  должны запускаться и тестироваться независимо друг от друга.
- **Логируйте каждый эпизод.** Инструкция, сид, выполненные подзадачи, результат,
  тайминги — итоговая статистика успеха собирается из этих логов.

### 6. Что нельзя решать в одиночку

Базовая политика, источник демонстрационных данных, распределение «рассуждения»
между политикой и отдельным LLM/VLM-слоем, целевая точность и устройство —
открытые командные вопросы (PRD §7). Выносите их на обсуждение, а не решайте
молча внутри PR.
