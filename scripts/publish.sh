#!/usr/bin/env bash
# Put the running demo on a public HTTPS address, from the rented box itself.
#
# Why not Cloudflare: the quick tunnel registers and then never receives a request. Tested on
# 2026-09-21 against an empty `python3 -m http.server` on the same machine, so it is not our
# application - the edge cannot deliver back through this host. Only one of the four tunnel
# connections ever registers, over QUIC and over http2 alike. A named tunnel would use the same
# transport, which is why no Cloudflare account was needed in the end.
#
# What works: a reverse SSH tunnel to localhost.run. No account, no domain, no token, and one
# long-lived TCP connection, which is apparently what this host's network will carry. HTTPS and
# WebSocket both pass - measured: page 200 in 1.0 s, socket open in 755 ms, round trip 255 ms
# from Kyiv.
#
# HTTPS is not a nicety here. A browser hands over a microphone only on a secure origin, so
# without this there is no demo at all.
#
#   ssh … 'bash -s' < scripts/publish.sh
#
# The hostname changes every time the tunnel restarts. It is a demo address, not an address to
# put in a document.
set -euo pipefail

PORT=${PORT:-8000}
LOG=${LOG:-/workspace/tunnel.log}

# pkill -f would match this script's own command line and kill the session running it; that has
# cost an afternoon more than once. Match the executable name exactly instead.
for pid in $(pgrep -x ssh || true); do kill "$pid" 2>/dev/null || true; done
sleep 2

# setsid --fork, because `nohup … &` over ssh intermittently does not survive the session.
setsid --fork bash -c "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 \
  -R 80:127.0.0.1:$PORT nokey@localhost.run > $LOG 2>&1 < /dev/null"

for _ in $(seq 1 20); do
  host=$(grep -aEo '[a-z0-9]+[.]lhr[.]life' "$LOG" 2>/dev/null | head -1 || true)
  [ -n "$host" ] && break
  sleep 3
done

if [ -z "${host:-}" ]; then
  echo "no hostname after 60s; see $LOG" >&2
  exit 1
fi

echo "https://$host/app/"
if [ -n "${MECHANIC_ACCESS_KEY:-}" ]; then
  echo "https://$host/app/?k=$MECHANIC_ACCESS_KEY   <- the link to send"
fi
