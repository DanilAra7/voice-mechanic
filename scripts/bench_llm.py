#!/usr/bin/env python3
"""Benchmark one LLM candidate behind llama-server: VRAM, raw speed, then the eval scenarios.

Every candidate is measured the same way, which is the whole point — the day-3 decision rests
on comparing these numbers, not on impressions.

    python scripts/bench_llm.py --name qwen3-30b-q3 \
        --model /workspace/models/Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf -- -c 16384 -ngl 99 --jinja
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

PROBE_PROMPT = "A driver says their temperature gauge is climbing. In three sentences, say what to do first."


def gpu_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


class VramSampler(threading.Thread):
    """Poll GPU memory in the background; llama-server allocates lazily, so the peak is what matters."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self._done = interval, 0, threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            try:
                self.peak = max(self.peak, gpu_mib())
            except Exception:
                pass
            self._done.wait(self.interval)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=3)
        return self.peak


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become ready")


def probe_speed(port: int, model: str, max_tokens: int = 200) -> dict:
    """One streamed completion: time to first token and steady-state tokens/s."""
    started = time.monotonic()
    first = None
    tokens = 0
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": True,
        },
        timeout=300,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                if first is None:
                    first = time.monotonic() - started
                tokens += 1
    elapsed = time.monotonic() - started
    gen = elapsed - (first or 0)
    return {
        "ttft_ms": round((first or 0) * 1000),
        "tokens": tokens,
        "tokens_per_s": round(tokens / gen, 1) if gen > 0 and tokens else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Label for this candidate in the results table")
    p.add_argument("--model", required=True, help="Path to the GGUF file")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--server", default="/workspace/llama/bin/llama-server")
    p.add_argument("--out-dir", default="evals/results")
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--skip-eval", action="store_true", help="Only measure VRAM and raw speed")
    p.add_argument("server_args", nargs="*", help="Extra llama-server flags after --")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.name}.server.log"

    baseline = gpu_mib()
    cmd = [args.server, "-m", args.model, "--port", str(args.port), "--host", "127.0.0.1", *args.server_args]
    print(f"[{args.name}] baseline VRAM {baseline} MiB\n[{args.name}] {' '.join(cmd)}", flush=True)

    sampler = VramSampler()
    sampler.start()
    load_started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(args.port, proc)
            load_s = round(time.monotonic() - load_started, 1)
            print(f"[{args.name}] ready in {load_s}s", flush=True)

            speed = probe_speed(args.port, args.name)
            print(f"[{args.name}] ttft {speed['ttft_ms']} ms, {speed['tokens_per_s']} tok/s", flush=True)

            if not args.skip_eval:
                eval_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "mechanic.evals.run",
                    "--model",
                    args.name,
                    "--base-url",
                    f"http://127.0.0.1:{args.port}/v1",
                    "--out",
                    str(out_dir / f"{args.name}.json"),
                ]
                if args.eval_limit:
                    eval_cmd += ["--limit", str(args.eval_limit)]
                subprocess.run(eval_cmd, check=False)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            peak = sampler.stop()

    summary = {}
    eval_path = out_dir / f"{args.name}.json"
    if eval_path.exists():
        summary = json.loads(eval_path.read_text())["summary"]
    bench = {
        "name": args.name,
        "model_path": args.model,
        "server_args": args.server_args,
        "vram_peak_mib": peak,
        "vram_baseline_mib": baseline,
        "vram_model_mib": peak - baseline,
        "load_seconds": load_s,
        "speed": speed,
        "eval": summary,
    }
    (out_dir / f"{args.name}.bench.json").write_text(json.dumps(bench, indent=2))
    print(f"\n[{args.name}] peak VRAM {peak} MiB (model {peak - baseline} MiB)")
    print(f"[{args.name}] wrote {out_dir / f'{args.name}.bench.json'}")


if __name__ == "__main__":
    main()
