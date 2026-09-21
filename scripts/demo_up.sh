#!/usr/bin/env bash
# Everything between a stopped instance and a lead talking to the mechanic, in one command.
#
# Run it on the rented box after `vastai start instance <id>`:
#
#   MECHANIC_ACCESS_KEY=… MECHANIC_NGROK_DOMAIN=… ssh … 'bash -s' < scripts/demo_up.sh
#
# Roughly ten minutes end to end, nearly all of it the models warming: the language model loads
# 13 GB, and the synthesiser pays CUDA warm-up plus priming the fixed lines on its first call.
# Do not let that happen while somebody is waiting - the first sentence of a cold process takes
# minutes, and it is the first thing they would hear.
set -euo pipefail

: "${MECHANIC_ACCESS_KEY:?set it - an open URL is strangers on our GPU}"
: "${MECHANIC_NGROK_DOMAIN:?set it to the reserved ngrok domain}"
VEHICLE=${VEHICLE:-audi_a4_b8}
FAULT=${FAULT:-vacuum_leak}

echo "== model and API"
/workspace/gpu_start.sh

# gpu_start.sh brings the API up without the gate; restart it with one. Killed by PID because
# `pkill -f` matches this script's own command line and would take the session with it.
for pid in $(pgrep -f "mechanic.server:app" | grep -v "^$$\$" || true); do kill "$pid" 2>/dev/null || true; done
sleep 4

cd /workspace/mechanic
export PATH=$HOME/.local/bin:$PATH
export MECHANIC_LLM_BASE_URL=http://127.0.0.1:8001/v1 MECHANIC_LLM_MODEL=gpt-oss-20b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid --fork bash -c "uv run uvicorn mechanic.server:app --host 0.0.0.0 --port 8000 > /workspace/api.log 2>&1"

for _ in $(seq 1 60); do
  curl -sf -o /dev/null http://127.0.0.1:8000/api/health && break
  sleep 3
done
echo "api ready"

echo "== a car on the ramp"
curl -sf -X POST "http://127.0.0.1:8000/api/sim/demo?k=$MECHANIC_ACCESS_KEY" \
  -H 'content-type: application/json' \
  -d "{\"vehicle\":\"$VEHICLE\",\"mode\":\"idle\",\"fault\":\"$FAULT\",\"time_scale\":10}" > /dev/null
echo "$VEHICLE running with $FAULT"

echo "== public address"
bash scripts/publish.sh

echo
echo "== warming the models so the lead does not pay for it"
# One throwaway conversation. The first sentence of a cold process is minutes; after this it is
# under a second, and the fixed lines are in the synthesiser's cache.
uv run python scripts/bench_voice.py --url ws://127.0.0.1:8000/ws/voice \
  --audio evals/audio/real --limit 2 --device warmup --out /tmp/warmup.json 2>&1 | tail -3
echo
echo "ready. send the link above."
