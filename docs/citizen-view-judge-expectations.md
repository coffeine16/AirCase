# The Citizen View, Read by a Judge

**What this is.** A gap analysis of `app/frontend/src/app/citizen/` written from the
chair of a finals judge who runs a company and invests — someone who will not be
impressed by a map, and who probes for *reach, retention, and whether a real person's
behaviour actually changes*. Grounded in reading the components, not the docs.

**Framing that matters:** the citizen view is already unusually strong on the axis the
project cares most about — **accountability**. `WardAccountability.tsx` answers *why is
my air bad*, *what is being done*, and *how sure are you*, which is genuinely rare.
The gaps below are almost all on a different axis: **reach and relevance**. The product
assumes a citizen who opens a web app and reads. The population it names as most
exposed — outdoor workers — is the population least likely to do that.

---

## What already exists (do not re-propose these)

| Built | Where |
|---|---|
| Ward finding — geolocate / tap map / search | `citizen/page.tsx` |
| Live AQI + category + 24h worsening/improving badge | `citizen/[wardId]/page.tsx` |
| **Why your air is like this** — attribution, evidence chain, confidence | `WardAccountability.tsx` |
| **What is being done** — case status + legal basis | `WardAccountability.tsx` |
| **How much to trust this** — km to nearest real monitor | `WardAccountability.tsx` |
| Advisory text (CPCB wording when real, generic fallback otherwise) | `citizen/[wardId]/page.tsx` |
| Voice advisory, 4 languages, **with per-language verification disclosure** | `VoiceAdvisory.tsx` |
| 72h forecast, 3-hourly, diurnal-aware, with 80% interval | `WardTimeline.tsx` |
| Report pollution (category + photo + description) | `citizen/[wardId]/report/page.tsx` |
| My reports, status badges | `citizen/reports/page.tsx` |
| Explore map — air / forecast / cases, 2D/3D, status-coloured pins | `citizen/explore/page.tsx` |

Two deliberate omissions in `WardAccountability.tsx` — no named private entity, no
SO₂/AAI evidence — are **correct and defensible**. Say so out loud when asked; they are
evidence of discipline, not gaps.

---

## Tier 1 — the questions a judge is most likely to actually ask

### 1. "How does this reach someone who never opens your website?"

**The gap.** Everything is pull. A citizen must remember AirCase exists, open a
browser, and find their ward. There is no push of any kind — no notification, no
WhatsApp, no SMS, no IVR call. `VoiceAdvisory` is real audio, but it plays from an
`<audio>` tag *on a page the citizen already had to open*, which assumes away the
exact barrier it exists to solve.

**Why it bites.** This is the single most likely investor question, and the project's
own architecture doc (Layer 6 / F6, F7) names it as core. A judge who reads the pitch
and then uses the product will notice the promise is unbuilt. It is also the retention
question in disguise: a civic app nobody reopens has no users, only visits.

**What closes it.** Genuinely: a WhatsApp/Telegram opt-in that pushes one message when
tomorrow crosses a threshold. The n8n outbound plumbing already exists (Layer 7) — this
is a subscribe button plus a row in a table, not new infrastructure.
**Cheap version for the demo:** a working "Notify me" opt-in that stores the preference
and shows the exact message that *would* be sent. Honest, and it proves the design.

### 2. "I have asthma. My neighbour is a 25-year-old cyclist. Why do we see the same screen?"

**The gap.** `AQI_ADVICE` in `lib/constants.ts` is keyed by AQI category alone. Every
reader gets identical text.

**Why it bites hardest here specifically.** Vulnerability is **20% of your EPS
formula**. The admin console explicitly weights schools, hospitals and eldercare — the
platform's equity claim rests on it. Yet the one surface an actual vulnerable person
touches treats everyone identically. A sharp judge will spot the inconsistency between
the pitch's equity story and the citizen product, and it undercuts the strongest
non-technical argument you have.

**What closes it.** One onboarding question — *outdoor worker / has a child / respiratory
condition / none* — stored in `localStorage` exactly as `ward_id` already is, used only
to re-word the advisory and the "worst around Xpm" line. No backend, no account, no new
data. Already scoped in `citizen-view-improvements.md` as "persona-lite advisory
filtering."

### 3. "AQI 403. What does that actually mean to me?"

**The gap.** The number is shown with a category label ("Severe") and generic advice.
There is no comparison that makes 403 legible to someone without a mental model.

**Why it bites.** Every AQI product in the world shows a number and a colour. Yours
differentiates on accountability but not on *comprehension*. A judge will ask what a
citizen is supposed to do with the figure, and "avoid prolonged exertion" is the same
sentence the government app already shows.

**What closes it** — any one of these, not all:
- **vs. the WHO guideline** ("27× the WHO 24-hour limit") — one line, real authority.
- **vs. your own city** ("worse than 84% of Delhi right now") — you already have the
  full fusion field client-side; this is a percentile over an array you have loaded.
- **vs. yesterday / last week** — the trend badge does 24h; the number needs memory.
- The cigarette-equivalent framing is famously effective and famously contested. If you
  use it, label it as an illustrative analogy, not a measurement — consistent with how
  the rest of the project handles uncertain claims.

### 4. "Your deck says 'your report led to an inspection.' Show me."

**The gap.** F7 loop closure is unbuilt. `citizen/reports/page.tsx` renders status
badges, but nothing connects a citizen's report to an actual inspection outcome. The
architecture doc calls this "the single most emotionally resonant beat available to the
demo" — and it is currently the one beat that does not exist.

**Why it bites.** You have named it as the climax yourself. If the demo narrates it and
the product cannot show it, that reads worse than never having claimed it.

**What closes it.** Full version needs the ledger + inspector reply loop. **Demo
version:** for a report whose ward has an `actioned`/`resolved` case in `actions.json`,
show the join — "a case in your ward moved to *inspected* on the 14th." That is a
client-side join over two files you already fetch, and it is *true*, not staged.

### 5. "How old is this number?"

**The gap.** No timestamp anywhere on the ward page. The AQI card, the advisory, and the
forecast all render without saying when the underlying data was computed.

**Why it bites.** The project's whole credibility posture is "we tell you what we don't
know" — monitor distance, confidence, withdrawn claims. Freshness is the one uncertainty
that is invisible, and it is the one a judge can catch instantly by asking. On a static
deploy reading precomputed JSON, the honest answer may be "several hours old," and
saying so *strengthens* the trust story rather than weakening it.

**What closes it.** One line under the AQI reading. `fusion.ts` is already in the
payload.

---

## Tier 2 — real gaps, less likely to be asked directly

**No history.** 24h trend and 72h forecast, but nothing backwards. The accountability
question a resident actually has is *"is my ward getting better or worse over the
year?"* — and you have two years of backfilled panel data per city sitting in
`data/historical/`. This is the clearest case where existing data would answer a
question the UI does not ask.

**No saved places.** One ward, stored in `localStorage`. People have a home, a
workplace, and a child's school, often in different wards. A judge with kids will ask.

**No escalation lever.** The product tells a citizen who is responsible and that
nothing is being done — then offers no action beyond reporting again. Naming a problem
without giving the citizen a lever is a slightly uncomfortable place to leave them.
Even a ward-officer contact or a "sign on to this ward's case" would close the arc.

**No sharing.** No way to send a ward's status to anyone. Civic products spread
socially; there is currently no mechanism for that at all.

**No privacy statement.** The app mints a device UUID and reads geolocation. Nothing
tells the citizen this, or that there is no account and no phone number. Given a
government-adjacent buyer, an investor will ask about the data-protection posture —
and the honest answer here is genuinely good, so it should be visible.

**Report submission sets no expectation.** After submitting, the citizen learns nothing
about what happens next or when. One sentence.

---

## What to say when the answer is "not built"

With two days, most of Tier 1 will not ship — and that is fine if the framing is right.
The project's strongest characteristic, everywhere else, is naming its own limits
precisely. Apply the same discipline here:

- **On reach:** *"Everything you see is pull. The push layer — WhatsApp and IVR — is
  designed, the channel plumbing exists, and it is not built. We would rather show you
  a working accountability chain than a fake notification."*
- **On personalization:** *"Vulnerability is 20% of our enforcement score, but the
  citizen view doesn't yet ask who you are. That is an inconsistency we know about."*
- **On the loop:** *"The inspector reply loop is the piece that turns this from a
  reporting form into a feedback system. It needs the ledger, which is specified and
  not built."*

A judge who has seen forty demos rewards a team that knows exactly what is missing far
more than one that pretends nothing is.

---

## Suggested order for the remaining two days

Ranked by *judge-visible impact per hour*, not by importance in the abstract.

| # | Item | Why first | Backend needed |
|---|---|---|---|
| 1 | Data freshness timestamp | Minutes of work; closes a credibility hole in the one place your trust story has a blind spot | No |
| 2 | Persona-lite advisory | Fixes the equity inconsistency, which is your most exposed argument | No |
| 3 | One comparison line (WHO or city percentile) | Makes the headline number mean something; pure client-side arithmetic | No |
| 4 | Report → case join ("a case in your ward was inspected") | Delivers the demo's emotional beat truthfully | No |
| 5 | "Notify me" opt-in that stores intent and shows the message | Answers the reach question with something real rather than a slide | Minimal |

Items 1–4 need no backend work at all, and every one of them is a display change over
data the citizen pages already fetch.

**Before any of this**, fix the blocker: `/citizen/[wardId]` and `/citizen/explore`
currently fail to build (missing `confidenceLabel`, `ACTION_STATUS_*` exports), and
hitting either one takes down every other route until the dev server restarts. None of
the above matters while the ward page itself will not render.
