# Deploy runbook — n8n on GCP with Terraform, Caddy & DuckDNS

**For the Keshav + Shyam call.** Follow top to bottom; every command is paste-ready.
Total time ~40 minutes, most of it waiting for the VM and the TLS cert.

**What you are building:** the project's ONE always-on box — an `e2-small` VM in
Mumbai running n8n (the channel layer) behind Caddy (automatic HTTPS), reachable at
a DuckDNS domain. It receives webhooks from **Telegram** AND from the **citizen
dashboard's report form**, and writes to Supabase. Everything else in the project
stays batch / static / scale-to-zero.

**Who does what on the call:**
| | |
|---|---|
| **Shyam** | Cloud Shell: Terraform, VM setup (he owns the GCP project) |
| **Keshav** | DuckDNS subdomain, Telegram BotFather, n8n workflows after |

**Have ready before 2:30:** Shyam logged into console.cloud.google.com (project
`aq-intelligence`), Keshav logged into duckdns.org (GitHub login works) and Telegram.

---

## Step 0 — sanity checks (Shyam, Cloud Shell, 2 min)

Open Cloud Shell (the `>_` icon, top right of the GCP console). Terraform is
**preinstalled and already authenticated** there — nobody installs anything locally.

```bash
gcloud config set project aq-intelligence
gcloud config get-value project        # must print: aq-intelligence
terraform version                      # any 1.x is fine
```

Confirm the billing budget alert still exists (Billing → Budgets & alerts → the $50
alert). If it is gone, recreate it — 1 minute, protects the $300.

---

## Step 1 — Terraform: create the infrastructure (Shyam, 10 min)

```bash
git clone https://github.com/coffeine16/EconomicTimes.git
cd EconomicTimes/deploy
terraform init
terraform plan     # READ THIS: it must say it will create exactly 4 resources:
                   # project_service, address, firewall, instance. Nothing destroyed.
terraform apply    # type: yes
```

At the end it prints:

```
static_ip = "34.xx.xx.xx"    <- WRITE THIS DOWN. Everything below uses it.
```

**If `apply` errors with "Compute Engine API has not been used… propagating":**
the API was enabled seconds ago and GCP is catching up. Wait 60 seconds, run
`terraform apply` again — Terraform resumes where it stopped; already-created
resources are untouched.

**If it errors on quota/billing:** check the project is `aq-intelligence` (step 0)
and billing is linked.

> State stays local in Cloud Shell (`deploy/terraform.tfstate` in the clone). It is
> gitignored — never commit it. Cloud Shell home persists between sessions, so
> `terraform destroy` from the same folder works later if ever needed.

---

## Step 2 — DuckDNS: point a name at the IP (Keshav, 3 min)

1. Go to **https://www.duckdns.org** and sign in.
2. Add subdomain: **`aq-intel`** (if taken, anything memorable — write down what you
   chose; it appears in three places below as `aq-intel.duckdns.org`).
3. In the row for your subdomain, paste the **static_ip** from step 1 into the
   `current ip` box and press **update ip**.

Verify from Cloud Shell:

```bash
nslookup aq-intel.duckdns.org    # must return the static_ip
```

If it returns nothing, wait 1–2 minutes and retry — DNS propagation for DuckDNS is
usually near-instant.

> Our IP is **static**, so this is a one-time setting. The DuckDNS-updater cron
> containers you see in tutorials exist for home connections with changing IPs —
> we do not need one, and the DuckDNS token never touches the VM or the repo.

---

## Step 3 — onto the VM (Shyam, 5 min)

```bash
gcloud compute ssh n8n --zone=asia-south1-a
```

(First time: it generates an SSH key — accept the prompts, empty passphrase is fine.)

The VM's startup script installs Docker; it can still be running when you first SSH
in. Check:

```bash
docker --version
```

**If "command not found":** the startup script is still working — it needs ~2–3
minutes after boot. Watch it finish with:

```bash
sudo journalctl -u google-startup-scripts -f    # Ctrl+C when it completes
```

Then confirm `docker --version` works.

---

## Step 4 — start n8n + Caddy (Shyam driving, Keshav confirming, 10 min)

Still inside the VM:

```bash
sudo git clone https://github.com/coffeine16/EconomicTimes.git /opt/aq
cd /opt/aq/deploy

# The ONE piece of config: which domain Caddy serves and n8n registers webhooks on.
# USE THE SUBDOMAIN KESHAV CREATED in step 2:
echo "N8N_DOMAIN=aq-intel.duckdns.org" | sudo tee .env

sudo docker compose up -d
```

Watch Caddy obtain the TLS certificate (~10–30 seconds):

```bash
sudo docker compose logs -f caddy
```

**Success looks like:** `certificate obtained successfully` for your domain. Ctrl+C.

**If the cert fails:**
- `NXDOMAIN` / DNS errors → step 2 didn't propagate yet, or the IP was mistyped in
  DuckDNS. Fix, then `sudo docker compose restart caddy`.
- `connection refused` on the challenge → the firewall didn't apply; from Cloud
  Shell run `gcloud compute firewall-rules list | grep n8n` (the rule
  `n8n-allow-http-https` must exist — Terraform created it).
- Rate-limit errors → Caddy automatically retries against ZeroSSL, its second CA.
  Give it 2 minutes before touching anything.

**Now open `https://aq-intel.duckdns.org` in a browser.** You get n8n's setup
screen.

> ⚠️ **Do this immediately, before anything else:** create the owner account (use
> the shared team email + a strong password in the team vault). Until the owner
> account exists, the editor is open to anyone who finds the URL.

---

## Step 5 — Telegram intake (Keshav, 5 min)

The workflow (**"Telegram citizen intake"**, already in the instance) uses a plain
**Webhook** node plus `$env.TELEGRAM_BOT_TOKEN`, *not* n8n's Telegram Trigger.
That is deliberate: the trigger node needs a stored **credential**, credentials
live only inside the `n8n_data` volume, and nothing outside that volume can
rebuild them. When the GCP box died it took the credential — and therefore this
workflow — with it. A token in compose is recreated by `docker compose up`.

1. Telegram → **@BotFather** → `/mybots` → pick the bot → **API Token**
   (or `/newbot` if you are making a fresh one). **Do not paste the token into
   chat, a commit, or a workflow node** — unlike the Supabase anon key this one
   is not public: it is full control of the bot.
2. Add it to `.env` **on the VM** (the same file that already holds
   `SUPABASE_URL` / `SUPABASE_ANON_KEY`):
   ```bash
   echo "TELEGRAM_BOT_TOKEN=<paste>" >> ~/aircase/.env
   cd ~/aircase && docker compose up -d      # recreates n8n with the new env
   ```
3. In n8n, open **Telegram citizen intake** → **Activate** (toggle, top-right).
4. Point Telegram at the webhook. n8n does this automatically *only* for the
   Telegram Trigger node, so with a plain webhook you register it once yourself:
   ```bash
   curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook?url=https://aq-intel.duckdns.org/webhook/telegram"
   # -> {"ok":true,"result":true,"description":"Webhook was set"}
   ```

**Test:** message the bot from your phone (“garbage being burned near the lake”).
The execution appears in n8n, a row lands in `citizen_reports`, and the bot
replies naming the category it filed.

**What it does.** Webhook → extract text/photo/voice + chat_id → one Gemini call
classifies into `waste_burning | construction | industrial | traffic | other`
with a one-line description and urgency → validated in **code**, never trusted
raw → insert into Supabase → acknowledge in the chat.

**The ward is deliberately left `null`.** An LLM cannot know which ward a message
came from, and a plausible wrong ward is worse than none — it files the report
against a place the citizen never mentioned, and attribution then counts it as a
corroborating instrument for the wrong cell. Ward is resolved downstream from a
real coordinate when Telegram sends one.

**If Gemini is down,** the report is still stored: the HTTP node is
`neverError` + `continueRegularOutput` and the parser falls back to the raw
message text with category `other`. Principle 2 — no LLM path goes silent.

**If the trigger never fires:** check `getWebhookInfo` (below) before anything
else — the usual cause is that step 4 was skipped or the workflow is saved but
not **Active**.
```bash
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo"
```

---

## Step 6 — citizen dashboard webhook (Keshav, 5 min)

The web report form is the second intake — same n8n, no Telegram involved.

1. New workflow → **Webhook** node:
   - Method `POST`, path `citizen-report`
   - In the node's **Options → Allowed Origins (CORS)**: `*` for now (tighten to the
     Vercel domain after frontend deploy).
2. **Activate.** The production URL becomes:

```
https://aq-intel.duckdns.org/webhook/citizen-report
```

3. Test from any terminal:

```bash
curl -X POST https://aq-intel.duckdns.org/webhook/citizen-report \
  -H "Content-Type: application/json" \
  -d '{"ward_id":"W218","category":"waste_burning","description":"test report","source":"web"}'
```

The execution appears in n8n. The frontend's report form then POSTs to this URL —
set it as `NEXT_PUBLIC_N8N_WEBHOOK_URL` in the frontend's Vercel env when deploying.

> The report payload written to Supabase must follow the **`citizen_corroboration`
> contract** (ward_id, category ∈ industrial|construction|waste_burning|traffic,
> ts, lat, lon, media_url, source) — attribution reads that exact shape.

> ⚠️ **`device_id` needs a mapping check.** The webhook body now also carries a
> `device_id` (an anonymous per-browser id, no PII) — it's how the citizen
> dashboard's "My reports" screen (`GET /reports?device_id=...`) shows a citizen
> only their own reports instead of every report in the city. `db/schema.sql`
> has the column. If the workflow's Supabase-insert node is set to auto-map
> incoming JSON fields, this needs nothing further; if it uses explicit
> field-to-column mapping, add `device_id -> device_id`. Test it: submit two
> reports with different fake `device_id` values via curl, then check in the
> Supabase table editor that the column is actually populated (not null) —
> without this mapping, reports still save, they just never surface in "My
> reports" for anyone, silently.

---

## Step 7 — end-of-call checklist

- [ ] `terraform output static_ip` matches what DuckDNS points at
- [ ] `https://<your-subdomain>.duckdns.org` loads n8n over valid HTTPS (padlock)
- [ ] n8n **owner account created** (editor is not open to the internet)
- [ ] Telegram bot message → execution visible in n8n
- [ ] `curl` to `/webhook/citizen-report` → execution visible in n8n
- [ ] Reboot survival: `sudo reboot`, wait 2 min, confirm the URL loads again
      (everything is `restart: unless-stopped` + Docker volumes — workflows and
      certs persist)

## Costs & care

- **e2-small ≈ $13/month** against the $300 credits. **Leave it running through
  judging** — stopping the VM kills both webhooks; this is the one box that must
  stay up. Tear everything down afterwards with `terraform destroy` (Cloud Shell,
  same folder).
- Workflows + credentials live in the `n8n_data` Docker volume, TLS certs in
  `caddy_data`. `docker compose down` keeps volumes; only `down -v` deletes them —
  **never pass `-v`** unless you mean to wipe n8n.

## What this deliberately does NOT do

- No SSL/DNS plugins, no DuckDNS token on the VM (static IP = set once).
- No n8n on Cloud Run (webhooks need always-on; scale-to-zero fights it).
- No pipeline on this box — the batch pipeline, GEE and evals run where they always
  did. This VM only moves bytes: **channels are dumb, agents are smart.**
