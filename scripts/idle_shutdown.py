#!/usr/bin/env python3
"""Stop the rented Vast.ai instance after a period with no conversation activity.

Renting by the hour means a forgotten instance keeps billing after the laptop is closed.
This watcher polls the backend's /api/health, which reports when the last user turn happened,
and stops the instance through the Vast.ai API once that exceeds --idle-minutes.

    CONTAINER_ID=$CONTAINER_ID VAST_API_KEY=... python scripts/idle_shutdown.py --idle-minutes 20

Dry run (prints instead of stopping): add --dry-run.
"""

import argparse
import os
import sys
import time

import httpx

VAST_API = "https://console.vast.ai/api/v0"


def last_activity_age_s(backend: str, timeout: float = 5.0) -> float | None:
    """Seconds since the backend last served a conversation turn, or None if unreachable."""
    try:
        r = httpx.get(f"{backend}/api/health", timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    if (age := data.get("seconds_since_activity")) is None:
        return None
    return float(age)


def stop_instance(instance_id: str, api_key: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"[dry-run] would stop instance {instance_id}", flush=True)
        return True
    r = httpx.put(
        f"{VAST_API}/instances/{instance_id}/",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"state": "stopped"},
        timeout=30,
    )
    print(f"stop instance {instance_id}: HTTP {r.status_code} {r.text[:200]}", flush=True)
    return r.is_success


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--idle-minutes", type=float, default=20)
    p.add_argument("--poll-seconds", type=float, default=60)
    p.add_argument("--instance-id", default=os.environ.get("CONTAINER_ID"))
    p.add_argument("--api-key", default=os.environ.get("VAST_API_KEY"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.instance_id or not (args.api_key or args.dry_run):
        print("Need --instance-id (or CONTAINER_ID) and --api-key (or VAST_API_KEY).", file=sys.stderr)
        return 2

    idle_limit = args.idle_minutes * 60
    unreachable_since: float | None = None
    print(f"watching {args.backend}; stopping {args.instance_id} after {args.idle_minutes} idle minutes", flush=True)
    while True:
        age = last_activity_age_s(args.backend)
        now = time.monotonic()
        if age is None:
            # A backend that is down is itself a reason to stop paying, but give it a grace period.
            unreachable_since = unreachable_since or now
            if now - unreachable_since > idle_limit:
                print("backend unreachable for the whole idle window", flush=True)
                return 0 if stop_instance(args.instance_id, args.api_key, args.dry_run) else 1
        else:
            unreachable_since = None
            if age > idle_limit:
                print(f"idle for {age / 60:.1f} minutes", flush=True)
                return 0 if stop_instance(args.instance_id, args.api_key, args.dry_run) else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
