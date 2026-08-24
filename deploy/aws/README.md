# Deploy runbook — n8n on AWS (replaces the GCP build)

The GCP project's billing is disabled pending KYC, which stopped **both** the
Compute VM running n8n **and** the Cloud Run API in one go — one cause, two
symptoms. Nothing in the stack was GCP-specific: `deploy/docker-compose.yml` and
`deploy/Caddyfile` are copied across unchanged.

**Earth Engine stays on Google and is unaffected.** It is a data source, not a
hosting choice, and its free non-commercial tier works regardless of the
project's billing state — verified, S5P queries still return.

## What Terraform already built

    cd deploy/aws && terraform init && terraform apply

| resource | value |
|---|---|
| region | `ap-south-1` (Mumbai) |
| instance | `t3.micro`, Debian 12, 20 GB gp3, Docker preinstalled |
| elastic IP | **3.108.47.243** |
| ports open | 80 (ACME), 443 (webhooks), 22 (SSH) |

The IP is *elastic* for the same reason the GCP build used a static address:
Telegram registers its webhook against a hostname that must keep resolving here,
and Caddy's certificate is issued for it. Replacing the instance does not move
the address.

## Step 1 — repoint DuckDNS  ← THE ONLY MANUAL STEP

`aq-intel.duckdns.org` still points at the dead GCP box (34.93.10.207).

1. Sign in at https://duckdns.org
2. Set the `aq-intel` subdomain's IP to **3.108.47.243**
3. Wait for it to take effect:

       nslookup aq-intel.duckdns.org      # must return 3.108.47.243

Do this **before** starting Caddy. Let's Encrypt issues via the HTTP-01
challenge, which requires the hostname to already resolve to this box; starting
early just burns failed validation attempts against the rate limit.

## Step 2 — start the stack

    ssh admin@3.108.47.243
    docker compose up -d
    docker compose logs -f caddy      # watch for "certificate obtained successfully"

`docker-compose.yml`, `Caddyfile` and `.env` (`N8N_DOMAIN=aq-intel.duckdns.org`)
are already on the box.

## Step 3 — n8n is a fresh install

The workflows lived in a Docker volume on the GCP VM and did **not** come across.
They have to be rebuilt in the n8n UI: the Telegram trigger, media download, the
one multimodal extraction call, ward validation, and the write to Supabase.

Supabase is separately gone — its project was reclaimed and the hostname no
longer resolves — so a new project is needed first, and its URL and anon key
have to be set in three places: n8n's credentials, the pipeline's `.env` (for
`scripts/sync_supabase.py`), and Vercel's environment.

## What is NOT blocked by any of this

The product itself. The frontend serves precomputed JSON contracts from
`app/frontend/public/data/`, which is why all eight cities render with the
channel layer down. What is missing while n8n is offline is citizen report
intake and the outbound advisory push — the read-only demo path is untouched.

`intelligence/agents/attribution.py` treats an absent channel layer as
`n_reports: 0` and leaves every score exactly as it was, so the pipeline's
numbers do not move either.
