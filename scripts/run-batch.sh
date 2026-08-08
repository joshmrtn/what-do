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
BATCH="${PROJECT_DIR}/.venv/bin/what-do-run-batch"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/batch-$(date +%Y%m%d-%H%M%S).log"

# flock releases the lock when this shell exits, however it exits.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "$(date -Is) another batch is still running; skipping this start" >>"${LOG_DIR}/batch-skipped.log"
    exit 0
fi

cd "${PROJECT_DIR}" || exit 1

# Point at this run before it starts, not after it ends. A batch here runs for
# hours, and pointing the link only on the way out meant that for the whole time
# anyone actually wanted to watch a run, the link named the *previous* one --
# which reads as a finished batch, complete with its summary and exit 0. Past
# the flock, so a skipped start still leaves the link on the live run.
ln -sfn "${LOG_FILE}" "${LOG_DIR}/batch-latest.log"

{
    echo "=== batch started $(date -Is) ==="
    echo "=== args: $* ==="
} >>"${LOG_FILE}"

PYTHONUNBUFFERED=1 "${BATCH}" "$@" >>"${LOG_FILE}" 2>&1
STATUS=$?

{
    echo "=== batch finished $(date -Is), exit ${STATUS} ==="
} >>"${LOG_FILE}"

exit "${STATUS}"
