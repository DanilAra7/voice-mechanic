#!/usr/bin/env bash
# Bring a rented GPU box from bare image to a working voice agent.
#
# Every flag here was paid for with measurement, so it lives in the repository rather than on a
# machine that gets destroyed: the model and the synthesiser share one 16 GB card and only fit
# with the KV cache quantised, a small compute batch, and PyTorch told not to fragment.
#
# Rent with: image nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04, disk 35 GB, --cancel-unavail.
# Ubuntu 24.04 is not optional: the prebuilt llama.cpp binaries need glibc 2.38.
#
#   ssh -p <port> root@<host> 'bash -s' < scripts/gpu_up.sh
#   # then, from the laptop:
#   git archive --format=tar HEAD | gzip -c | ssh … 'cd /workspace/mechanic && tar xzf -'
#   tar czf - data/processed data/index | ssh … 'cd /workspace/mechanic && tar xzf -'
#   ssh … /workspace/gpu_start.sh
set -euo pipefail

LLAMA_BUILD=${LLAMA_BUILD:-b11037}
WORK=/workspace
MODELS=$WORK/models
ASR_MODELS="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

echo "== system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# python3.12-dev is not optional: without the headers Triton cannot build its CUDA helper and
# the synthesiser dies on its first sentence.
apt-get install -y -qq curl libgomp1 ca-certificates python3.12-dev
curl -LsSf https://astral.sh/uv/install.sh | sh > /tmp/uv.log 2>&1
export PATH=$HOME/.local/bin:$PATH

echo "== auto-stop after ${DEADMAN_HOURS:-4}h"
# Uses the key Vast gives the container, so the account key never lands on a rented machine.
cat > /root/deadman.sh <<'DEADMAN'
#!/bin/bash
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
sleep "$1"
curl -s -X PUT "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
  -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" \
  -d '{"state":"stopped"}'
DEADMAN
chmod +x /root/deadman.sh
setsid nohup /root/deadman.sh $(( ${DEADMAN_HOURS:-4} * 3600 )) > /root/deadman.log 2>&1 < /dev/null &

echo "== llama.cpp $LLAMA_BUILD"
mkdir -p "$WORK/llama" "$MODELS"
cd "$WORK/llama"
for f in "llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz" \
         "cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
  curl -sL -o "$f" "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/$f"
  tar xzf "$f" && rm "$f"
done
# The runtime libraries ship in the second archive and have to sit beside the binaries.
cp "cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64"/*.so* "llama-$LLAMA_BUILD/"
ln -sfn "$WORK/llama/llama-$LLAMA_BUILD" "$WORK/llama/bin"

echo "== models (~13 GB)"
cd "$MODELS"
[ -f gpt-oss-20b-MXFP4.gguf ] || curl -sL -C - -o gpt-oss-20b-MXFP4.gguf \
  "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-MXFP4.gguf"
[ -d sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 ] || {
  curl -sL -o asr.tar.bz2 "$ASR_MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
  tar xjf asr.tar.bz2 && rm asr.tar.bz2
}
[ -f silero_vad_v5.onnx ] || curl -sL -o silero_vad_v5.onnx "$ASR_MODELS/silero_vad_v5.onnx"
mkdir -p "$WORK/mechanic/data/models"
ln -sfn "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8" "$WORK/mechanic/data/models/"
ln -sfn "$MODELS/silero_vad_v5.onnx" "$WORK/mechanic/data/models/"

echo "== run scripts"
cat > "$WORK/gpu_start.sh" <<'START'
#!/usr/bin/env bash
# Start both halves. The synthesiser loads into whatever the model leaves free, so the model's
# appetite is capped deliberately: q8_0 KV cache and a 128-token compute batch buy ~200 MiB,
# which is the difference between running and an out-of-memory on the second question.
set -euo pipefail
export LD_LIBRARY_PATH=/workspace/llama/bin
pgrep -f llama-server > /dev/null || setsid nohup /workspace/llama/bin/llama-server \
  -m /workspace/models/gpt-oss-20b-MXFP4.gguf --port 8001 --host 127.0.0.1 \
  -c 8192 -ngl 99 --jinja --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 512 --ubatch-size 128 \
  --chat-template-kwargs '{"reasoning_effort":"low"}' \
  > /workspace/llm.log 2>&1 < /dev/null &

for _ in $(seq 1 90); do curl -sf http://127.0.0.1:8001/health > /dev/null && break; sleep 2; done
echo "model ready"

cd /workspace/mechanic
export PATH=$HOME/.local/bin:$PATH
export MECHANIC_LLM_BASE_URL=http://127.0.0.1:8001/v1
export MECHANIC_LLM_MODEL=gpt-oss-20b
# Fragmentation, not size, is what runs the card out of room when both models are resident.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pgrep -f "uvicorn mechanic" > /dev/null || setsid nohup \
  uv run uvicorn mechanic.server:app --host 0.0.0.0 --port 8000 \
  > /workspace/api.log 2>&1 < /dev/null &

for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8000/api/health > /dev/null && break; sleep 2; done
echo "api ready on :8000  (the browser client is at /app/)"
START
chmod +x "$WORK/gpu_start.sh"

echo
echo "done. upload the code and data, then run: $WORK/gpu_start.sh"
