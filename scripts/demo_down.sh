#!/usr/bin/env bash
# Stop paying for the GPU.
#
#   ./scripts/demo_down.sh
#
# STOP, not destroy. A stopped instance keeps its disk for $0.019/h with every model already on
# it; reinstalling from scratch costs about $0.85 in bandwidth, so stopping pays for itself
# after roughly 45 idle hours. The ngrok domain is reserved, so the link is the same next time.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "no .env" >&2; exit 1; }
set -a; . ./.env; set +a
: "${VAST_INSTANCE_ID:?set it in .env}"

vastai stop instance "$VAST_INSTANCE_ID" >/dev/null
echo "asked instance $VAST_INSTANCE_ID to stop"

for _ in $(seq 1 30); do
  state=$(vastai show instances --raw | python3 -c 'import json,sys,os
wanted = os.environ["VAST_INSTANCE_ID"]
print(next((i["actual_status"] for i in json.load(sys.stdin) if str(i["id"]) == wanted), "missing"))')
  case "$state" in
    running|loading) sleep 5;;
    *) break;;
  esac
done

echo "state: $state"
[ "$state" = "running" ] && { echo "STILL RUNNING - check console.vast.ai yourself" >&2; exit 1; }
echo "the GPU is no longer billing. The link will work again after ./scripts/demo.sh"
