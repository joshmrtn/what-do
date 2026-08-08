#!/usr/bin/env bash
# Nightly batch wrapper.
#
# Exists because cron gives a job almost no environment and no memory of the
# last one. Three things it guarantees:
#
#   * a lock, so a run that is still going is never joined by another. On this
#     hardware a batch takes many hours, which is longer than the interval
#     between some scheduling mistakes.
#   * a dated log per run, kept where a person will look for it.
#   * the project's own venv and working directory, neither of which cron knows.
#
# Usage: run-batch.sh [extra what-do-run-batch flags...]

set -uo pipefail

PROJECT_DIR="/home/ubuntu/projects/what-do"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="/tmp/what-do-batch.lock"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/batch-$(date +%Y%m%d-%H%M%S).log"

# flock releases the lock when this shell exits, however it exits.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date -Is) another batch is still running; skipping this start" >>"${LOG_DIR}/batch-skipped.log"
    exit 0
fi

cd "${PROJECT_DIR}" || exit 1

{
    echo "=== batch started $(date -Is) ==="
    echo "=== args: $* ==="
} >>"${LOG_FILE}"

"${PYTHON}" -u -c 'import sys; from src.scheduler import run; sys.exit(run())' "$@" \
    >>"${LOG_FILE}" 2>&1
STATUS=$?

{
    echo "=== batch finished $(date -Is), exit ${STATUS} ==="
} >>"${LOG_FILE}"

# Leave the newest run easy to find without knowing today's date.
ln -sfn "${LOG_FILE}" "${LOG_DIR}/batch-latest.log"

exit "${STATUS}"
