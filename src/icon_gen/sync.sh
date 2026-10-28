#!/bin/bash
# Back up the icon_gen working set to Google Drive, or restore it on another machine:
#   pixi run icons-sync push    # local -> Drive
#   pixi run icons-sync pull    # Drive -> local
# Both directions copy and never delete, so pushing from an empty or half-restored
# working set cannot wipe the backup. The review marks (reviewed.json) are the part
# that cannot be regenerated; the pictures can, at the cost of allowance.
#
# Needs an rclone remote named `gdrive`, authorised once in a browser (README). Its
# `drive.file` scope lets rclone see only the files it created, not the rest of Drive.
set -euo pipefail
rclone listremotes | grep -qx 'gdrive:' || {
    echo "no rclone remote 'gdrive': run  pixi run rclone config create gdrive drive scope=drive.file" >&2
    exit 1
}
REMOTE="gdrive:icon_gen"
# Drive's free 15 GB is shared with Gmail and Photos, so a push past 8 GB is refused
# rather than left to fill the account.
LIMIT_BYTES=$((8 * 1024 ** 3))
case "${1:-}" in
    push)
        size=$(du -sb --exclude sd-server.log "$ICON_GEN_OUT" | cut -f1)
        if (( size > LIMIT_BYTES )); then
            echo "refusing: $ICON_GEN_OUT is $((size / 1024 ** 2)) MB, over the 8 GB guard" >&2
            exit 1
        fi
        rclone copy "$ICON_GEN_OUT" "$REMOTE" --exclude sd-server.log --fast-list --progress ;;
    pull) rclone copy "$REMOTE" "$ICON_GEN_OUT" --fast-list --progress ;;
    *) echo "usage: pixi run icons-sync push|pull" >&2; exit 2 ;;
esac
