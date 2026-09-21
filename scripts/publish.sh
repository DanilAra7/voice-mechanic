#!/usr/bin/env bash
# Put the running demo on a public HTTPS address, from the rented box itself.
#
# HTTPS is not a nicety: a browser hands over a microphone only on a secure origin, so without
# this there is no demo at all.
#
# Why not Cloudflare: the quick tunnel registers and then never receives a request. Tested on
# 2026-09-21 against an empty `python3 -m http.server` on the same machine, so it is not our
# application - the edge cannot deliver back through this host, over QUIC and http2 alike, with
# one of four connections ever registering. A named tunnel uses the same transport, which is why
# no Cloudflare account was needed in the end.
#
# What works is a single long-lived outbound connection. Three were tried:
#
#   ngrok        stable name, socket open 196 ms, round trip 71 ms   <- this
#   serveo       new name on every reconnect, 218 ms, 59 ms
#   localhost.run new name on every reconnect, 755 ms, 255 ms
#
# The names matter more than the milliseconds. Both free SSH tunnels reassign the hostname when
# their session reconnects, and the session reconnects silently: the link dies in the hands of
# the person it was sent to while every check on this side still passes. It happened twice.
# ngrok's free tier gives one reserved domain, which is the whole point.
#
# Its free tier also puts an interstitial in front of the first HTML load - one click, then a
# cookie. The WebSocket and the API are not affected.
#
#   ssh … 'bash -s' < scripts/publish.sh
#
# Needs: ngrok installed, `ngrok config add-authtoken …` run once, and MECHANIC_NGROK_DOMAIN set.
set -euo pipefail

PORT=${PORT:-8000}
LOG=${LOG:-/workspace/ngrok.log}
DOMAIN=${MECHANIC_NGROK_DOMAIN:?set MECHANIC_NGROK_DOMAIN to the reserved ngrok domain}

# pkill -f would match this script's own command line and kill the session running it; that cost
# four attempts in one afternoon. Match the executable name exactly instead.
for pid in $(pgrep -x ngrok || true); do kill "$pid" 2>/dev/null || true; done
sleep 2

# A loop, not a command: a free tunnel drops, and it drops silently.
# setsid --fork, because `nohup … &` over ssh intermittently does not survive the session.
setsid --fork bash -c "while true; do
  ngrok http $PORT --domain=$DOMAIN --log=stdout >> $LOG 2>&1
  echo \"--- ngrok exited, restarting \$(date -u +%H:%M:%S) ---\" >> $LOG
  sleep 3
done"

for _ in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$DOMAIN/api/health" || true)
  [ "$code" = "200" ] && break
  sleep 3
done

if [ "${code:-}" != "200" ]; then
  echo "tunnel did not answer after 60s; see $LOG" >&2
  exit 1
fi

echo "https://$DOMAIN/app/"
if [ -n "${MECHANIC_ACCESS_KEY:-}" ]; then
  echo "https://$DOMAIN/app/?k=$MECHANIC_ACCESS_KEY   <- the link to send"
fi
