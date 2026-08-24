# Spec — Citizen View Improvements (Layer 9)

**Correction to the previous version of this doc:** it assumed "frontend not
built," following `CLAUDE.md`/`architecture.md`, which are stale on this
point. `app/frontend/src/app/citizen/` is a real, fairly sophisticated
Next.js citizen dashboard — ward finding, live AQI, a "why is my air like
this" evidence panel, voice advisories, a 72h timeline, and pollution
reporting are all built. This version is grounded in reading that code, not
the docs. **Whoever owns `CLAUDE.md` should update its "NOT built: ...
frontend" line — it's no longer true for the citizen side.**

---

## What's actually built today

| feature | where | notes |
|---|---|---|
| Ward finding (geolocate / tap map / search) | `citizen/page.tsx`, `useWardLocator.ts` | client-side H3 resolve, no server round-trip |
| Live AQI card + 24h trend badge | `citizen/[wardId]/page.tsx` | reads the fusion field directly |
| **"Why your air is like this"** — attribution, evidence, confidence | `WardAccountability.tsx` | this *is* Tier 1 from the last version of this doc, already done, and done carefully — see below |
| **"What is being done"** — action status + legal basis | `WardAccountability.tsx` | reads `actions.json` / memo legal citations, joined by zone not ward |
| **"How much to trust this number"** — distance to nearest real monitor | `WardAccountability.tsx` | says outright when a reading is model-estimated vs. station-anchored |
| Text advisory (CPCB text when available, generic fallback otherwise) | `citizen/[wardId]/page.tsx` | doesn't double-render — generic advice is suppressed when a real one exists |
| **Voice advisory (F6)**, per language, with a verification disclosure | `VoiceAdvisory.tsx` | already labels machine-written vs. native-reviewed vs. CPCB-official text — this was Tier 0 in the old doc and it's done, including the honesty framing that doc asked for |
| 72h forecast, 3-hourly, diurnal-aware | `WardTimeline.tsx` | deliberately not a Now/24/48/72 strip — explicit comment explains why that hid the day/night swing |
| Report a pollution source (category + photo + description) | `citizen/[wardId]/report/page.tsx` | routes through n8n → Supabase per the Layer 7 design (channel is dumb) |
| My reports list, with status badges | `citizen/reports/page.tsx` | **see the gap below — this currently always renders empty** |
| Theme toggle, multi-city switcher | `ThemeToggle.tsx`, `CitySwitcher.tsx` | |
| **Explore map** — air quality / forecast / cases, one map, switchable | `citizen/explore/page.tsx`, `CitizenMap.tsx` | built this session — see below, this section is now historical |

`WardAccountability.tsx` in particular is worth reading in full — it has two
deliberate omissions (no named private entity, no SO2/AAI evidence surfaced)
that are exactly the anti-overclaiming discipline `architecture.md` asks for
elsewhere, applied to the one surface where the audience is a member of the
public instead of an inspector.

---

## The explore map — built this session

The original version of this section proposed `/citizen/actions`: a citywide,
read-only analog of the admin's map + `ActionQueue` panel, since "what is
being done" previously only existed as a paragraph inside
`WardAccountability.tsx`, scoped to whichever single ward a citizen happened
to be looking at. It's now built, at **`/citizen/explore`**, and ended up
broader than the original proposal in one way and different in another:

- **Broader:** it isn't just cases — it's one map with three switchable modes
  (`CitizenExploreMap`'s chip switcher: Air quality / Forecast / Cases), since
  a citizen exploring the city wants all three, not just the enforcement
  layer. Air quality and Forecast reuse the same fusion-field and
  forecast-endpoint data the ward page already reads; Cases is the part that
  maps to the original proposal.
- **Different interaction model:** instead of a scrollable list of cards
  beside the map (the admin `ActionQueue` pattern), tapping a hex or pin
  surfaces a single detail card *below the map*, and tapping elsewhere
  replaces it. No side panel, no list to scroll — the map itself is the
  primary surface, which is closer to "explore" than "triage queue," and
  reads better on a phone.

**What it withholds, exactly as specced:** no dispatch routes, no raw EPS
score, no named private entity, no memo-generation controls. Pin colour is
**status**, not severity — `pending → dispatched → actioned → resolved`,
labelled in plain language via `ACTION_STATUS_LABELS`
(`lib/constants.ts`), not the admin's internal vocabulary. Confidence is
shown via `confidenceLabel()` (also new this session — the "bucket
confidence into plain language" item from Tier 1 below), not a raw float.

**No new backend work was needed**, as predicted: `actions.json` and
`memos.json` were already being fetched successfully elsewhere
(`WardAccountability.tsx`), confirming this really was a frontend-only gap.

**What's still open:**
- **Sort/filter for the Cases mode.** Right now every open case shows as a
  pin; there's no way to filter by status or category, and with more than a
  handful of cases the map alone (no list) may need a count or a "N cases in
  view" affordance. Not built.
- **`CitizenMap.tsx` now renders two different layer types** (H3 hexes via
  `cells`, or point pins via the new `pins` prop) depending on which prop is
  passed, sharing one deck.gl instance, theme observer, and fit-to-data
  effect. If a fourth mode is ever added (e.g. FIRMS fire markers), extend
  this same prop-driven pattern rather than forking a new map component.

**What to deliberately withhold, and why** — same anti-overclaim discipline
`WardAccountability.tsx` already applies elsewhere, extended to two things
that are new risks specifically because this view is public and citywide:
- **No dispatch routes.** `DispatchLayer.tsx` draws literal per-team road
  paths and stop sequences from `dispatch.json`. Publishing that tells
  anyone — including whoever is about to be inspected — exactly which order
  and road an inspector is travelling right now. This is an operational-
  security problem, not a UX one; it should never reach a public page.
- **No raw EPS score or its component breakdown.** `EPSBreakdown` in
  `ActionQueue.tsx` is a triage tool for deciding dispatch order; showing a
  citizen "your ward scored 43/100" invites "why isn't mine first" about a
  number that was never designed to justify itself publicly.
- **No named private entity, no "Generate memo" / dispatch controls.** Those
  stay inspector-only, for the reason `WardAccountability.tsx`'s own comment
  already states: 0/4 recall on NO2-confounded sources means naming a
  specific business publicly would be an accusation the evidence doesn't
  support.

---

## `/reports` — fixed this session, with one external dependency left open

**1. Report photos are still captured and then silently dropped.**
`report/page.tsx` builds a `photo` preview and includes `photo` in
`CreateReportPayload`. But `api.ts::submitReport` posts only
`{ ward_id, category, description, lat, lon, device_id, source }` to the n8n
webhook — `payload.photo` is never read, never attached, never sent. **Not
touched this session** — still open, still the same bug.

**2. `/reports` now exists and is wired end to end on the repo side.** What
changed:
- `db/schema.sql`: added a `device_id` column to `citizen_reports` (plus an
  `alter table ... add column if not exists` migration, since the
  `create table if not exists` at the top of the file is a no-op against an
  already-provisioned Supabase instance — editing it alone would never have
  reached the live database) and an index on it.
- `app/backend/main.py`: a real `GET /reports?device_id=...` route. It reads
  Supabase directly over the same REST call `scripts/sync_supabase.py`
  already makes (no new dependency — `urllib`, matching that script's
  style), filtered server-side by `device_id` before anything reaches a
  browser. Degrades to `[]` on any failure (unset Supabase env vars,
  unreachable host, etc.) rather than 500ing — same "a dead channel must not
  break the request" discipline the rest of the API already follows.
- `useReports.ts`: the `// Q2 open` identity question is resolved — a new
  `useDeviceId()` hook mints a `crypto.randomUUID()` on first use and keeps
  it in localStorage, same pattern as `useCitizenWard()`. No account, no
  phone number. `api.ts::submitReport` now sends it with every report;
  `api.ts::getReports` now requires it.

**What's still an open dependency, and I can't close it from here:** the
report form POSTs to an **external** n8n instance (a live VM, not code in
this repo — see `deploy/README.md`). Adding `device_id` to the JSON body
guarantees nothing until n8n's Supabase-insert step actually maps that field
into the new column; if the workflow uses explicit field mapping rather than
auto-mapping incoming JSON, someone with n8n access needs to add one row.
Documented as a flagged step in `deploy/README.md` (Step 6) with a concrete
test (submit via curl with a fake `device_id`, check the Supabase table
editor for a non-null value) — but I have no credentials or tool access to
the n8n instance itself, so this needs a person to actually go verify it.
Until then, reports keep saving fine; they just won't have a `device_id` and
so won't surface in anyone's "My reports."

**Also not verified: the backend code itself.** No Python environment with
`fastapi` installed was available this session (no `pip`, on this machine) —
`app/backend/main.py` passed a static syntax check (`ast.parse`) and the
Supabase REST call mirrors `sync_supabase.py`'s already-working pattern
closely, but the endpoint has not actually been run against a live Supabase
instance. Worth an `uvicorn app.backend.main:app --reload` + a curl before
trusting it in front of a demo.

---

## Real improvements — refinements on top of what exists

**Confidence is still a raw percentage, not plain language.** `Why your air is
like this` shows `confidence 67%`. `eval_attribution.py` validated specific
thresholds (100% precision above 0.70; 0.66 median on hits vs. 0.42 on
misses) — bucket the same number into "high confidence" / "likely" /
"under review" using those thresholds, so a non-technical reader isn't left
guessing whether 67% is good or bad. This is a display change over data that
already exists; `WardAccountability.tsx` is the one file to touch.

**Persona-lite advisory filtering.** `AQI_ADVICE` is keyed by category only,
same generic text for every reader. A single client-side onboarding
question — outdoor worker / has a child / respiratory condition / none —
stored the same way `ward_id` already is (localStorage, no account), used
only to re-word `AQI_ADVICE` and the "worst around Xpm" line already
computed in `WardTimeline.tsx`. No backend change; templating over data
that's already there.

**"Check it yourself" links, where the attribution names a real, checkable
event.** The Bhalswa result is only powerful because a citizen can
independently verify it. `WardAccountability.tsx` already restricts itself
to a category + zone, correctly, given 0/4 NO2-confounded recall — that
restraint should stay. But where an attribution's evidence includes a real
FIRMS fire-detection window, surfacing the dates lets a skeptical citizen
search for it themselves, the same way the doc's own Delhi headline is
meant to be checked.

**Report photo → also feeds the report-status screen.** Once (1) above is
fixed, the photo becomes real evidence — worth showing the citizen's own
submitted photo back to them on `citizen/reports/page.tsx` (the `ReportCard`
component already has a `photo_url` render path, it just has nothing to
render yet).

**IVR / WhatsApp-first reach**, still genuinely unbuilt. `VoiceAdvisory`
plays audio through a `<audio>` tag on a webpage — real, but it still
assumes a citizen opened the web app. The brief's own framing (Layer 6/F6)
is that the most-exposed population skews low-literacy and may never load
this page. That gap is unchanged by anything above; it's a telephony/channel
integration, not a UI one.

**F5 silent-ward weighting**, still unbuilt — no ward covariates (internet
penetration, literacy) found anywhere in the codebase. Once `/reports` is
live, this becomes checkable in a way it currently isn't: you could compare
report volume per ward against `WardAccountability`'s hotspot findings and
see, empirically, which wards are "severe but silent."

---

## Reversed: 3D map, and the explore page going full-bleed

The previous version of this doc argued against 3D/extrusion for the citizen
view, reasoning it was an admin-console fit, not a "stay sparse" citizen one.
The user overrode that directly — asked for the explore map specifically to
be bigger, extruded, and full-page like the admin console — so it's built:

- `CitizenMap.tsx` gained two opt-in props, **off by default**: `extruded`
  (H3 hex layer gets `getElevation` from PM2.5, `elevationScale: 12`, and the
  camera opens at `pitch: 45, bearing: -12` instead of top-down) and `bleed`
  (drops the border/corner-radius for an edge-to-edge fill). Pins never
  extrude — only the choropleth does.
- Only `/citizen/explore` passes `extruded bleed height="100%"`. The
  ward-picker map (`citizen/page.tsx`) and the small ward-preview map
  (`citizen/[wardId]/page.tsx`) are unchanged — still flat, bordered,
  fixed-height — so the original reasoning still holds *for those two*; it
  just doesn't apply to a dedicated full-page map the way it applies to a
  small embedded one.
- `extruded` isn't fixed on — citizens get a live **2D/3D toggle** (top-right
  glass chip pair). Flipping it re-tilts the camera too (`pitch: 45` / `0`),
  not just the hex heights, via a new effect in `CitizenMap.tsx` that watches
  `extruded` after mount instead of only reading it once at init. Hidden in
  Cases mode, since pins never extrude — there'd be nothing for the toggle to
  visibly do.
- The explore page itself is now full-bleed: `height: calc(100vh - 52px)`
  (52 must track the citizen header's height in `citizen/layout.tsx`),
  `overflow: hidden`, and everything that used to sit below the map in normal
  document flow — the mode switcher, the forecast horizon picker, the
  legend, and the selected ward/case detail — is now a floating overlay
  (glass chips top-center, a glass bottom sheet for detail/legend). No page
  title either, matching the admin console's own map page, which has none.

**Not verified in a live browser** — no browser tool available this session.
`tsc --noEmit` and `npm run build` both pass, and the interaction logic
(fit-to-pins, extrusion elevation formula, sheet content switching) was
reasoned through, but the actual 3D visual — pitch angle, elevation scale,
whether the bottom sheet overlaps content awkwardly on a small phone
viewport — needs an eyeball pass in `npm run dev` before calling it settled.

---

## Suggested order

1. ~~**Send the photo that's already being captured**~~ — **still open**, not
   done this session. `api.ts::submitReport` still drops `payload.photo`.
2. ~~**Decide the device-identity model, then build `GET /reports`**~~ —
   **done, repo-side.** `useDeviceId()`, `db/schema.sql`'s `device_id`
   column, and `GET /reports` are all built. **One external step remains and
   isn't mine to close:** the live n8n workflow must map the new `device_id`
   field into Supabase — flagged in `deploy/README.md` Step 6 with a test to
   run. Also unverified: no Python env with `fastapi` was available this
   session, so the endpoint has only been syntax-checked, never run.
3. ~~**Bucket confidence into plain language**~~ — **done.** `confidenceLabel()`
   added to `lib/constants.ts` and wired into `WardAccountability.tsx` (now
   reads e.g. "High confidence (67%)") and into the new explore map's case
   cards.
4. ~~**Build a citywide map + cases view**~~ — **done, as `/citizen/explore`**
   (broader than originally scoped — see "The explore map" section above).
5. **Persona-lite advisory text** — still open, client-side only, no backend
   change needed.
6. **F7 loop closure** — repo-side blocker (2) is cleared; still needs the
   ledger + inspector WhatsApp "done" reply (`architecture.md` Layer 6/F7),
   which is a separate, larger build.
7. **IVR / WhatsApp-first reach, F5 ward covariates** — still open, larger
   parallelizable infra work.

What's left with no backend/external dependency: #1 (one line in `api.ts`)
and #5. Everything else either needs the n8n mapping check (part of #2) and
a real backend smoke-test, the ledger build (#6), or is larger infra (#7).
