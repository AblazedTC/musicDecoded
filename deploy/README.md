# Deploying to one server

The whole backend (API, Postgres, Redis, MinIO, worker) runs on one Ubuntu 24.04 server with Docker
Compose, behind Caddy for HTTPS. A 2 vCPU / 4 GB server runs one worker (~150-200 songs/hour).

## First time
1. Create the server (e.g. a DigitalOcean Droplet, Ubuntu 24.04, 4 GB) with your SSH key.
2. Point two DNS names at its IP, e.g. `api.example.com` and `files.example.com`. Without a domain,
   use `api.<ip>.sslip.io` and `files.<ip>.sslip.io` (dashes or dots in the IP both work).
3. On the server, as root:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/AblazedTC/musicDecoded/main/deploy/setup.sh | bash -s -- api.example.com files.example.com
   ```
   It generates the secrets into `/opt/musicdecoded/.env` (never committed) and prints `Up: https://...`
   when the API answers. Build the apps against `https://<api host>`.

Optional settings (YouTube search key, RevenueCat webhook secret, daily limits) go in
`/opt/musicdecoded/.env`; run `deploy/update.sh` after editing it.

## Updates
```bash
ssh root@<server> /opt/musicdecoded/deploy/update.sh
```

## Backups and logs
- Turn on the provider's server backups (DigitalOcean: Droplet → Backups). They cover the database and
  stored files.
- Logs: `cd /opt/musicdecoded && docker compose -f docker-compose.yml -f deploy/compose.prod.yml logs -f api worker`
