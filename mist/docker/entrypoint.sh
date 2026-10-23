#!/bin/sh
# Runs the app as PUID:PGID (Unraid default 99:100 = nobody:users) so files in /data are not root-owned.
set -e
PUID="${PUID:-99}"
PGID="${PGID:-100}"
umask "${UMASK:-022}"

getent group "$PGID" >/dev/null 2>&1 || groupadd -g "$PGID" mist
getent passwd "$PUID" >/dev/null 2>&1 || useradd -u "$PUID" -g "$PGID" -d /data -s /bin/sh mist

mkdir -p /data /data/cache /data/models /data/class /data/jobs
chown "$PUID:$PGID" /data /data/cache /data/models /data/class /data/jobs
# first start after an upgrade from root-owned files
[ "$(stat -c %u /data/models)" = "$PUID" ] || chown -R "$PUID:$PGID" /data

cd /app
exec gosu "$PUID:$PGID" "$@"
