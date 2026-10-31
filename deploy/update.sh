#!/usr/bin/env bash
# Deploy the latest main. On the server: /opt/musicdecoded/deploy/update.sh
set -euo pipefail
cd "$(dirname "$0")/.."
git pull --ff-only
git submodule update --init --recursive
docker compose -f docker-compose.yml -f deploy/compose.prod.yml up -d --build --remove-orphans
docker image prune -f >/dev/null
