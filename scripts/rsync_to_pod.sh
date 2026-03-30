#!/bin/bash

# --- ALOHA RunPod Data Sync Script ---
# Usage: ./rsync_to_pod.sh [pod_id] [ssh_port]

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 [pod_id] [ssh_port]"
    echo "Example: $0 xxxxxxxx 12345"
    exit 1
fi

POD_ID=$1
PORT=$2
REMOTE_USER="root"
# Note: You'll need to update the hostname based on your RunPod region
REMOTE_HOST="ss-${POD_ID}.runpod.net" 

echo "[Sync] Syncing local cache/ and code to RunPod workspace..."

# 1. Sync cache (crucial for 5GB data)
rsync -avzP -e "ssh -p $PORT" ./cache/ $REMOTE_USER@$REMOTE_HOST:/workspace/cache/

# 2. Sync weights
rsync -avzP -e "ssh -p $PORT" ./checkpoints/ $REMOTE_USER@$REMOTE_HOST:/workspace/checkpoints/

echo "[Sync] Done!"
