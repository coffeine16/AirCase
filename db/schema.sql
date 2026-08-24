-- Supabase schema — THE two shared contracts between n8n (writer) and the
-- intelligence pipeline (reader). Paste this whole file into the Supabase SQL
-- editor and run it once.
--
-- These shapes are LOAD-BEARING on the reader side:
--   citizen_reports   -> attribution's citizen_corroboration evidence term
--   inspection_status -> the ledger's response-time chain (already proven: the
--                        ledger consumed a hand-written file of this exact shape
--                        and computed response_hours with zero code change)
-- Change a column name here and the pipeline reads nothing. Extend freely with
-- NEW columns; never rename these.

-- ── 1. Citizen reports (web form + Telegram, via n8n) ────────────────────────
create table if not exists public.citizen_reports (
  id          uuid primary key default gen_random_uuid(),
  ts          timestamptz not null default now(),
  ward_id     text not null,            -- validated IN CODE by n8n against /wards
  category    text not null check (category in
                ('industrial','construction','waste_burning','traffic','other')),
  description text,
  lat         double precision,
  lon         double precision,
  media_url   text,                     -- Supabase storage URL for photo/voice
  source      text not null default 'web' check (source in ('web','telegram','whatsapp')),
  language    text,                     -- 'hi' | 'kn' | 'ta' | 'en' (for the reply)
  chat_id     text,                     -- Telegram chat to close the loop with
  device_id   text,                     -- anonymous per-BROWSER id, web reports only
                                         -- (Telegram reports are matched by chat_id
                                         -- instead). Minted client-side with
                                         -- crypto.randomUUID(), no account, no PII —
                                         -- it exists ONLY so GET /reports can return
                                         -- one citizen's own reports instead of every
                                         -- report in the city (RLS below lets anon
                                         -- read all rows; the API filters by this).
  status      text not null default 'submitted' check (status in
                ('submitted','under_review','corroborated','action_taken','resolved'))
);

-- Migration for an ALREADY-PROVISIONED instance: `create table if not exists`
-- above is a no-op once the table exists, so a column added only to that
-- statement never reaches a live database. Safe to re-run this whole file.
alter table public.citizen_reports add column if not exists device_id text;
create index if not exists citizen_reports_device_id_idx
  on public.citizen_reports (device_id);

-- ── 2. Inspector loop (the "done" reply -> the ledger's stopwatch) ───────────
create table if not exists public.inspection_status (
  action_id     text primary key,        -- matches actions.json action_id
  dispatched_at timestamptz,
  actioned_at   timestamptz,
  status        text not null default 'dispatched' check (status in
                  ('dispatched','actioned','resolved')),
  inspector     text,
  updated_at    timestamptz not null default now()
);

-- ── RLS: anon key may INSERT reports and READ both tables. Nothing may UPDATE
--         or DELETE via anon — status changes go through n8n's service role. ──
alter table public.citizen_reports  enable row level security;
alter table public.inspection_status enable row level security;

create policy "anon_insert_reports" on public.citizen_reports
  for insert to anon with check (true);
create policy "anon_read_reports" on public.citizen_reports
  for select to anon using (true);
create policy "anon_read_inspections" on public.inspection_status
  for select to anon using (true);
