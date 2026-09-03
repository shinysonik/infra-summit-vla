#!/usr/bin/env bash
#
# One-shot hook installation. Run once after cloning:
#
#     ./scripts/setup-hooks.sh
#
# It points git at the versioned .githooks/ directory, so the pre-push guard
# (and any future hook) is shared through the repo instead of living only in
# each contributor's .git/hooks/.
#
# Одноразовая установка хуков. Запустите один раз после клонирования.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true

echo "[EN] Hooks installed: core.hooksPath -> .githooks"
echo "[RU] Хуки установлены: core.hooksPath -> .githooks"
echo
echo "[EN] Direct pushes to the protected branch will now be blocked locally."
echo "[RU] Прямые пуши в защищённую ветку теперь блокируются локально."
