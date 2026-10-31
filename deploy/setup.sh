#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server, run as root:
#   curl -fsSL https://raw.githubusercontent.com/AblazedTC/musicDecoded/main/deploy/setup.sh | bash -s -- <api-host> <files-host>
# Installs Docker, adds swap and a firewall, clones the repo to /opt/musicdecoded, generates secrets
# into /opt/musicdecoded/.env (kept on the server only) and starts the stack.
set -euo pipefail
API_HOST=${1:?usage: setup.sh <api-host> <files-host>}
FILES_HOST=${2:?usage: setup.sh <api-host> <files-host>}
DIR=/opt/musicdecoded

command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh

# The worker peaks at ~1.4 GB and its image build needs more; swap keeps a 4 GB server from running out.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable

[ -d "$DIR/.git" ] || git clone --recurse-submodules https://github.com/AblazedTC/musicDecoded.git "$DIR"
cd "$DIR"

if [ ! -f .env ]; then
  (umask 077; cat > .env <<ENV
API_HOST=$API_HOST
FILES_HOST=$FILES_HOST
POSTGRES_PASSWORD=$(openssl rand -hex 24)
MINIO_ROOT_USER=md$(openssl rand -hex 4)
MINIO_ROOT_PASSWORD=$(openssl rand -hex 24)
JWT_SECRET=$(openssl rand -hex 48)
YOUTUBE_API_KEY=
REVENUECAT_WEBHOOK_SECRET=
ENV
  )
fi

deploy/update.sh
for i in $(seq 1 30); do curl -fs "https://$API_HOST/health" && { echo; echo "Up: https://$API_HOST"; exit 0; }; sleep 10; done
echo "API not reachable at https://$API_HOST yet; check: docker compose -f docker-compose.yml -f deploy/compose.prod.yml logs caddy api" >&2
exit 1
