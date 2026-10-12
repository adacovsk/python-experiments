#!/bin/bash
# Keep the Cloudflare side of the set going across daily allowances: run until the
# allowance is spent, then retry every 15 minutes. A refused first request costs
# nothing, and Cloudflare's reset does not land exactly at 00:00 UTC, so polling
# catches it whenever it happens. Flags from the live gallery are served first.
cd "$(dirname "$0")/../.."
source "$HOME/.profile"  # CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
while true; do
    # The small sets first (each is a fraction of one day's allowance), then abilities;
    # anything already rendered is cached, so later passes skip straight past them.
    python src/icon_gen/build.py --backend cloudflare --records conditions --all
    python src/icon_gen/build.py --backend cloudflare --records effects --all
    python src/icon_gen/build.py --backend cloudflare --all --watch
    echo "[$(date -u +%H:%M) UTC] allowance spent or run ended; retrying in 15 minutes"
    sleep 900
done
