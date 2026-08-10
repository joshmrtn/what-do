#!/usr/bin/env bash
# Consistent database snapshot.
#
# `cp` is not a backup once WAL is on: recent commits live in event_hub.db-wal,
# so copying the main file alone captures a torn state. VACUUM INTO takes a
# consistent snapshot of a live database and writes a single compact file with
# no sidecars, which is safe to run while a batch is writing.
#
# Usage: backup-db.sh [destination-dir]

set -euo pipefail

PROJECT_DIR="/home/ubuntu/projects/what-do"
DB="${PROJECT_DIR}/database/event_hub.db"
DEST="${1:-/home/ubuntu/what-do-backups}"
KEEP=7

if [ ! -f "${DB}" ]; then
    echo "no database at ${DB}" >&2
    exit 1
fi

mkdir -p "${DEST}"
SNAPSHOT="${DEST}/event_hub-$(date +%Y%m%d-%H%M%S).db"

sqlite3 "${DB}" "VACUUM INTO '${SNAPSHOT}'"

# A snapshot nobody has opened is a guess, not a backup.
if ! sqlite3 -readonly "${SNAPSHOT}" "PRAGMA integrity_check;" | grep -qx "ok"; then
    echo "integrity check failed for ${SNAPSHOT}" >&2
    exit 1
fi

echo "${SNAPSHOT}"

# Keep the newest KEEP snapshots; older ones are noise, and the database is
# rebuilt from candidates anyway if every copy is somehow lost.
ls -1t "${DEST}"/event_hub-*.db 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
    rm -f "${old}"
done
