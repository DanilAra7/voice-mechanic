#!/usr/bin/env bash
# From a stopped instance to a link somebody can talk to, in one command.
#
#   ./scripts/demo.sh
#
# About ten minutes, nearly all of it models loading. Run it BEFORE anyone opens the tab: the
# first sentence of a cold process takes minutes and is the first thing they would hear.
#
# Reads VAST_INSTANCE_ID, MECHANIC_ACCESS_KEY and MECHANIC_NGROK_DOMAIN from .env, which is
# gitignored - so no secret is ever typed into a shell, a chat window or a screen share.
#
# Safe to run twice. Everything it does is idempotent: an instance already running is left
# running, the API and the tunnel are restarted, and the warm-up happens again.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "no .env - copy .env.example and fill it in" >&2; exit 1; }
set -a; . ./.env; set +a
: "${VAST_INSTANCE_ID:?set it in .env}"
: "${MECHANIC_ACCESS_KEY:?set it in .env}"
: "${MECHANIC_NGROK_DOMAIN:?set it in .env}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "the box"
state=$(vastai show instances --raw | python3 -c 'import json,sys,os
wanted = os.environ["VAST_INSTANCE_ID"]
print(next((i["actual_status"] for i in json.load(sys.stdin) if str(i["id"]) == wanted), "missing"))')
case "$state" in
  missing) echo "instance $VAST_INSTANCE_ID does not exist any more - see docs/GPU.md to rent another" >&2; exit 1;;
  running) echo "already running";;
  *) echo "starting ($state)"; vastai start instance "$VAST_INSTANCE_ID" >/dev/null;;
esac

for _ in $(seq 1 40); do
  state=$(vastai show instances --raw | python3 -c 'import json,sys,os
wanted = os.environ["VAST_INSTANCE_ID"]
print(next((i["actual_status"] for i in json.load(sys.stdin) if str(i["id"]) == wanted), "missing"))')
  [ "$state" = "running" ] && break
  sleep 10
done
[ "$state" = "running" ] || { echo "instance never came up (last state: $state)" >&2; exit 1; }

# The SSH port is reassigned on every start, so it is read now rather than remembered.
url=$(vastai ssh-url "$VAST_INSTANCE_ID")
host=${url#ssh://root@}; port=${host##*:}; host=${host%%:*}
echo "ssh root@$host -p $port"

ssh_box() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p "$port" "root@$host" "$@"; }
for _ in $(seq 1 30); do ssh_box true 2>/dev/null && break; sleep 5; done

say "the code"
# git archive, not a deploy key on a rented machine. COPYFILE_DISABLE stops macOS tar from
# adding ._ resource forks that break the extraction on Linux.
COPYFILE_DISABLE=1 git archive --format=tar HEAD | gzip -c | ssh_box 'cd /workspace/mechanic && tar xzf -'
echo "$(git rev-parse --short HEAD) shipped"

say "model, API, car, tunnel, warm-up"
ssh_box "MECHANIC_ACCESS_KEY='$MECHANIC_ACCESS_KEY' MECHANIC_NGROK_DOMAIN='$MECHANIC_NGROK_DOMAIN' bash -s" \
  < scripts/demo_up.sh

cat <<EOF

────────────────────────────────────────────────────────────────
  Send this link:

  https://$MECHANIC_NGROK_DOMAIN/app/?k=$MECHANIC_ACCESS_KEY

  Click "Visit Site" once, then "Connect", then allow the microphone.
  A car is already running, so there is nothing to set up.

  When it is over:  ./scripts/demo_down.sh
────────────────────────────────────────────────────────────────
EOF
