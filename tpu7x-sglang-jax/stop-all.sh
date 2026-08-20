#!/bin/bash
# Stop the server on both nodes.
cd "$(dirname "$0")"
source ./env.sh

ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE1" \
    "cd $SCRIPT_DIR && ./stop.sh" || true
./stop.sh || true
