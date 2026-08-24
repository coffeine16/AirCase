<p align="center">
  <img src="./assets/banner.svg" alt="AirCase Banner" />
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-hackathon%20prototype-43C6F9?style=flat-square" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11-134C87?style=flat-square" />
  <img alt="models" src="https://img.shields.io/badge/models-LightGBM%20%2B%20robust%20stats-134C87?style=flat-square" />
  <img alt="spatial" src="https://img.shields.io/badge/spatial-H3%20res--8-134C87?style=flat-square" />
  <img alt="agents" src="https://img.shields.io/badge/agents-9--node%20LangGraph-134C87?style=flat-square" />
  <img alt="serving" src="https://img.shields.io/badge/serving-FastAPI%20%2B%20n8n%20channels-134C87?style=flat-square" />
  <img alt="data" src="https://img.shields.io/badge/data-Sentinel--5P%20%2F%20FIRMS%20%2F%20OpenAQ-43C6F9?style=flat-square" />
  <img alt="llm" src="https://img.shields.io/badge/LLM-explains%2C%20never%20ranks-43C6F9?style=flat-square" />
</p>

**An air quality platform that names *who* is polluting, *where*, with *what
evidence*, and *what to do about it today* — not another map of how bad the air
is.**

India does not have a monitoring problem; it has an action problem. Over 900
CAAQMS stations exist, and the *money* exists too — it is the action that does
not. A CAG performance audit tabled in the Delhi Assembly found **₹843.12 crore
of the ₹1,624.63 crore** collected as Environment Compensation Charge (2015-16
to 2024-25) still **unspent**, with collection records missing for **280 days**
and no explanation offered when CAG asked. Nationally, the 15th Finance
Commission released **₹8,357.63 cr** in air quality grants to 42 million-plus
cities and only **₹5,465.49 cr** was utilised — Bengaluru, released **₹535.10
cr**, utilised **₹0.00** ([PIB/Lok Sabha, 5 Feb 2024, Annexure-I](https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2002614)).
Cities are short of neither monitors nor budget; they are short of a defensible
answer to *where to act* — and **CAG has named that exact gap**. Report No. 2 of
2022 carries a section headed *"Lack of real-time information regarding sources
of pollutants"*:

> *"AQI values only indicate the total concentration of pollutants in an area.
> The source of each pollutant is measured by source apportionment/emission
> inventory study."*

Delhi paid **₹87.60 lakh** for such a study, terminated the contract in December
2020, and had **not finalised any study as of March 2021**. The same audit
concluded that without it, *"impact and success of any measure to control
pollution cannot be assessed."* **That study is what this platform produces
continuously, from free public data.**

That audit is archived in this repository at
[`docs/sources/`](./docs/sources/) rather than cited by URL alone — a claim whose
only evidence is a link is one dead link away from being unsourced, and this one
carries the argument.

Meanwhile the monitors themselves
lie about coverage: CPCB siting norms deliberately place them *away from
sources*, so the official map is a **measurement log, not a pollution census**.
That is not only our reading — the same CAG audit physically verified 13 of
Delhi's 24 DPCC stations and found **every one** of them non-compliant with CPCB
siting norms (trees on multiple sides; Anand Vihar and Wazirpur adjacent to
heavy-traffic roads), concluding this renders **"the AQI values unreliable"**.
None of the 24 stations measured Lead at all.

This platform detects sources from instruments that cover **every cell equally**,
names them with an inspectable evidence chain, and reports honestly on what it
**cannot** see.

<p align="center">
  <img src="./docs/img/console-map.png" alt="AirCase enforcement console: a Delhi hotspot zone selected, showing its attributed source, evidence chain, EPS breakdown and a routed inspection" width="100%" />
</p>

<p align="center">
  <sub><b>The enforcement console.</b> A chronic zone in KAPASHERA, attributed to
  traffic, with the deterministic score, the evidence behind it, the EPS
  breakdown, a routed inspection and a one-click memo. Every layer on the left is
  a real instrument.</sub>
</p>

<p align="center">
  <img src="./docs/img/citizen-ward.png" alt="AirCase citizen view for PERAGHARHI: AQI 403 Severe, with a panel explaining that no source was attributed here" width="31%" />
  &nbsp;
  <img src="./docs/img/citizen-forecast.png" alt="Same ward scrolled: the nearest monitor is 6.7 km away, the advisory, and a 3-hourly forecast naming the cleanest and worst hours" width="31%" />
  &nbsp;
  <img src="./docs/img/citizen-report.png" alt="The citizen report form: pollution type, optional photo, optional description" width="31%" />
</p>

<p align="center">
  <sub><b>The same platform, facing the citizen.</b> AQI 403 is the number every
  app shows. The three panels under it are the ones nobody else does.</sub>
</p>

**Why your air is like this.** Here it says *no source was attributed in your
ward* — because none was. Our instruments found nothing standing out, and the
panel says so rather than inventing a cause. Where a source *is* found, its
evidence chain appears in the same place.

**How much to trust this number.** *"The nearest government monitor is 6.7 km
away. There is no sensor in your ward, so this number is an estimate built from
satellite data, weather and local land use — not a measurement. Most of this city
has no monitor; showing you a number without saying so would be the dishonest
part."* Where a monitor **is** close, the same panel says so instead. The claim
tracks the evidence.

**When to go outside, and how to talk back.** Cleanest around 10am at AQI 271,
worst around 7pm at 383 — the swing across one day is larger than the change from
one day to the next, which is exactly what a 24-hour forecast hides. And the
report form turns a resident into the one instrument that sees construction dust,
which no satellite can.

---

## Run it in under a minute

No API keys. No cloud account. The whole pipeline runs offline against a
synthetic world with known ground truth, so every number below is reproducible on
your machine right now.

```bash
pip install -r requirements.txt
$env:PYTHONPATH = "."                                    # PowerShell (bash: export PYTHONPATH=.)

python scripts/run_pipeline.py --synthetic --full        # ingest → panel → fusion → detect → attribute
python scripts/eval_detection.py                         # THE headline stat
```

The `--full` run drives the whole **9-agent pipeline** and writes every JSON
contract (`hotspots`, `attributions`, `forecast`, `actions`, `dispatch`, `memos`,
`advisories`, `ledger`) to `data/outputs/`. Then serve it:

```bash
uvicorn app.backend.main:app --reload --port 8000        # 24 endpoints: /hotspots, /attribution/{cell}, /actions, /dispatch, /ledger, /advisories …
```

> **All three tiers are deployed.** The Next.js console and citizen view are live
> on Vercel; the read-only API and the 9-agent chain run on AWS behind Caddy; the
> channel layer (n8n citizen bot + Telegram intake) shares that box on the same
> certificate. See
> [what's real vs prototype](#whats-real-vs-prototype) — we'd rather tell you than
> let you find out.

---

## The headline result

**On real data.** Delhi, November 2025 — real Sentinel-5P, real NASA FIRMS, real
CPCB stations, real OpenStreetMap:

> ### Bhalswa landfill → `waste_burning`, confidence 0.84
> *0.40 km from the mapped landfill · evidence: satellite fire detections in
> 30 hours (18% of the window); shallow boundary layer trapping emissions*

A real polluter, in a real city, from public satellite data, with an evidence chain
**anyone can check** — google *"Bhalswa landfill fire November 2025"*.

**Then it did it again, in a different city, unprompted:**

> ### Pirana landfill, Ahmedabad → `waste_burning`, confidence 0.78, `chronic`
> *0.36 km · evidence: satellite fire detections in 51 hours (7% of the window);
> municipal landfill 3.0 km, wind alignment 0.47*

84 hectares, in use since 1982, a long public record of fires. Ahmedabad has **55
FIRMS detections in 90 days** — the most of any city we run, which is precisely
*why* fire-driven detection works there and not everywhere. Bhalswa was not a
fluke; it was the instrument working where the instrument can see.

(Okhla landfill → `waste_burning`, confidence 0.50, 1.00 km — it previously came
back as `traffic`, defensible since it sits on Mathura Road, but incomplete; that
weak spot has closed. Ghazipur → still not detected: no fires in the window. We
report all four.)

**On the synthetic world**, where ground truth exists and accuracy can actually be
*scored*, recall is reported by **what the instruments can physically see**:

| tier | recall |
|---|---|
| **direct — thermal fire** (waste burning) | **2/2 found and correctly named** — and **both appear on no map at all** |
| **NO₂, confounded** (industrial, traffic) | **0/4** — NO₂ is a real tracer, but the road network lifts it citywide, so a point source must out-shout its own neighbourhood |
| **no tracer at all** (construction) | **0/3** — coarse PM. Nothing sees it. |

Enforceable-**zone** precision **2/2**. Attribution **100%** (16/16, all
unregistered). ⚠️ Small n — never sell these as rates.

And the number the whole thing exists for:

> **An unbiased 12-monitor network catches a median of 1 source in 9.**
> It misses eight. That is not siting bias — that is *geometry*. A dozen sensors
> cannot cover a city, however honestly you place them.

### We used to claim 4/4. It was not real.

The two *industrial* sources were being found via **SO₂ contrast**. Then we pulled
the real satellite: **real S5P SO₂ over a city is noise** — 49% of readings are
*negative*, a physical impossibility, with a MAD 30× the median (SNR 0.7). TROPOMI's
SO₂ band is built for volcanoes; an urban factory is far below its floor. The
aerosol index is no better (SNR 1.0).

On real Delhi, scoring across all three channels flagged **470 of 1,703 cells — 28%
of the city — and 87% of those were driven by SO₂ or AAI noise.** We were
manufacturing enforcement targets out of retrieval error, and a genuinely burning
landfill ranked *below* them.

So we deleted both channels, and the industrial sources went with them. **4/4 → 2/4.**
Our simulation had given the instruments a signal they do not have — the same class
of error as [the 100% trap](#the-100-trap), one level deeper.

### The evaluation was flattering our own baseline

Forecasts are scored against **persistence** — *"it will stay as it is."* We kept
losing to it at 24 h and could not work out why the model was so weak.

It was not. The **evaluation** was.

PM2.5 runs on a daily cycle, so persistence is accidentally *in phase* at 24-hour
multiples — it compares a value to the same hour of the previous day. Between
those points it compares night to day and collapses. On real Delhi:

| lead | 3 h | 6 h | 12 h | **24 h** | **48 h** | **72 h** |
|---|---|---|---|---|---|---|
| our model | 81.5 | 85.2 | 87.2 | 88.3 | 91.2 | 92.1 |
| persistence | **67.2** | 91.9 | 105.8 | **92.6** | 110.5 | 115.5 |
| skill | **−21.3%** | +7.2% | +17.5% | **+4.6%** | **+17.5%** | **+20.3%** |

Read the persistence row, not ours. It climbs to 105.8 at 12 h and then *falls
back* to 92.6 at 24 h — the baseline gets **better** as the horizon gets longer,
which no forecast should. That is the diurnal cycle: at 24-hour multiples
persistence compares a value to the same hour of the previous day and is
accidentally in phase. Between those points it compares night to day and
collapses.

Every standard benchmark — including the one in our own brief — samples 24, 48
and 72 h. **Exactly where the baseline is strongest**, and at 24 h our margin is
its narrowest (+4.6%) for that reason.

**And we lose at 3 h, by 21%.** At that range "the air is what it is right now"
is genuinely hard to beat in Delhi, and we report it rather than starting the
table at 6 h. The horizon that matters for enforcement is 48–72 h — *"stagnant
winds Thursday, act before"* — and that is where the margin is widest.

⚠️ These are **out-of-sample**: 175,585 station-cell rows from 2025-11-30 to
2026-01-14, strictly after the model's training cutoff, with climatology
truncated at the same cutoff so it cannot leak either. Measured *inside* the
training window the same model "beats" persistence by 33–60%. That number is
fake, we found it, and we threw it away.

**Where we lose has moved, and we report the new one.** This paragraph used to
say we lost in Chennai until 51 h — true of the earlier single-city model.
Re-scored out-of-sample against the pooled quantile model, Chennai is now the
*strongest* city (+10% at 3 h rising to +27% at 72 h) and **Delhi is the one we
lose in, at 3 h**. All three, honestly:

| skill vs persistence | 3 h | 24 h | 48 h | 72 h |
|---|---|---|---|---|
| Delhi | **−21.3%** | +4.6% | +17.5% | +20.3% |
| Chennai | +10.1% | +12.4% | +21.2% | **+27.1%** |
| Bengaluru | +3.2% | +8.4% | +13.0% | +12.1% |

The model wins **3 of 3 cities at 24 h, 48 h and 72 h**. A method that only
publishes the city where it wins is not a method — so the one red cell stays.

We also tested adding a meteorology forecast, as an **oracle**: feeding the model
the met that actually occurred, the most favourable case that exists. It did not
help in any city, so we did not build the plumbing. A negative result we went
looking for.

---

## How it works — the whole system at a glance

Three tiers: a **batch data platform** turns raw feeds into a feature table; a
**9-agent pipeline** turns that table into named sources, priorities, memos and
advisories; a **read-only serving layer** exposes precomputed JSON that the
frontend and channels consume. Heavy compute never touches a request handler.

```mermaid
flowchart LR
    subgraph SRC["① Ingestion — 6 free sources, live + synthetic fallback"]
        direction TB
        OAQ["OpenAQ · CPCB stations"]
        S5P["Sentinel-5P · NO₂/SO₂/aerosol"]
        FIR["NASA FIRMS · thermal fires"]
        MET["Open-Meteo · wind + BLH"]
        OSM["OpenStreetMap · industry, roads"]
        CIT["Citizens · Telegram + web<br/>(n8n → Supabase)"]
    end

    SRC --> PANEL["② Panel<br/>H3 cell × hour feature table"]
    PANEL --> FUSION["Fusion field · LightGBM"]
    PANEL --> AGENTS

    subgraph AGENTS["③ Agent pipeline (LangGraph · 9 nodes)"]
        direction LR
        DET["detect"] --> ATT["attribute"] --> FC["forecast"] --> PR["prioritise"] --> MO["memo"] --> AD["advise"] --> VO["voice"] --> LG["ledger"] --> AU["audit"]
    end

    FUSION -->|"EXPOSURE · what people breathe"| API
    AGENTS -->|"SOURCES · who to inspect, what to do"| API
    API["④ Read-only FastAPI<br/>precomputed JSON contracts"] --> FE["Frontend · Next.js + deck.gl<br/>(deployed)"]
    API --> CH["Channels · n8n<br/>advisories, inspector loop"]

```

**LLMs explain. Deterministic code decides.** Category scores, priority scores,
and every ranking are plain reproducible arithmetic. An LLM may only write prose
explaining a score that was already computed — and if it disagrees with the
arithmetic, **the arithmetic wins and the LLM output is discarded.** Every
LLM path has a rule-based fallback with an identical output schema, so a missing
API key degrades the *prose*, never the *answer*. **This discipline holds even at
the edge** — the citizen-intake bot lets an LLM *extract* fields from a free-text
report, but deterministic code canonicalises the ward against the official list
and clamps the category before anything is stored (see the [citizen loop](#the-citizen-loop--from-a-phone-to-the-evidence-chain)).

---

## The agent pipeline — signal to action

One `LangGraph` state machine runs nine agents over a shared typed state. Each
node is wrapped so one failure degrades the output instead of killing the run; the
same graph backs both the batch pipeline and the API's `POST /run/agent`, so chain
order can never drift between them.

```mermaid
flowchart TD
    D["detect<br/><small>NO₂ contrast + fire persistence → zones</small>"]
    A["attribute<br/><small>deterministic category scores + evidence chain + confidence</small>"]
    F["forecast<br/><small>24–72 h PM2.5 vs persistence baseline</small>"]
    P["prioritise<br/><small>Enforcement Priority Score + dispatch routing</small>"]
    M["memo<br/><small>dispatch-ready notice + rule-matched legal basis</small>"]
    V["advise<br/><small>ward advisories · English + the city language<br/>hi · ta · kn</small>"]
    L["ledger<br/><small>freeze counterfactual forecast · track signal→memo→dispatch time</small>"]
    VO["voice<br/><small>advisory text → MP3 per ward, per language</small>"]
    AU["audit<br/><small>monitoring blind spots → next-sensor placement</small>"]
    D --> A --> F --> P --> M --> V --> VO --> L --> AU

    A -. "citizen reports<br/>corroborate" .-> CIT["citizen evidence<br/><small>from the channel layer</small>"]
    CIT -. "+1 independent instrument<br/>(capped)" .-> A

    classDef det fill:#0f6b3f22,stroke:#0f6b3f;
    classDef llm fill:#a9760a22,stroke:#a9760a;
    class D,P,L,AU det
    class A,F,M,V,VO llm
```

Green nodes are pure arithmetic; amber nodes call an LLM **for prose only**, each
with a rule-based fallback. Every agent writes a versioned JSON contract to
`data/outputs/` that the API serves read-only.

---

## Detection: why not the fusion field?

This is the most important thing in the repo, and it cost us a rewrite to learn.

The original design ran hotspot detection on the fusion field. **It cannot work,
and we measured why:**

```
Training stations see a mean source contribution of  0.25 µg/m³   (p99 = 6.7)
The rest of the city reaches                       210    µg/m³
Only 6 of 4,032 station-hours have a fire nearby — against 4,414 citywide
```

The model trains only on cells containing a station, and CPCB siting deliberately
places stations *away from sources* — the very fact the fusion layer's rationale
rests on. So it **never observes a source**, cannot learn a source response, and
(being a tree ensemble) **cannot extrapolate to one**: LightGBM predicts
piecewise-constant, so a cell whose NO₂ column is far above anything a station
ever saw gets the same prediction as the worst station. Its field is
background-dominated by construction.

**The fusion field is an exposure map, not a detector.** It answers *"what is a
person in this cell breathing"*. On the synthetic world it does that well — LOSO
R² 0.90, ~36% better than the station-mean map. **On real Delhi it does not**: it
is worse than a naive city mean, and we withdraw the claim rather than quote the
synthetic number. Detection is the contribution; see the status table.

Detection instead runs on the two instruments with **uniform coverage — every
cell, no siting bias:**

```mermaid
flowchart TD
    SAT["Satellite NO₂ column<br/>per-cell MEDIAN per window<br/>(SO₂/AAI dropped — measured noise)"] --> CON["Neighbourhood contrast<br/>vs the 4–8 km annulus, in MAD units"]
    FIRE["FIRMS detections<br/>fraction of window burning within 1.5 km"] --> CON
    CON --> W{"Multi-window agreement<br/>24h / 7d / 30d"}
    W -->|"elevated over 30 d"| CHR["chronic<br/>a standing violator → build the case file"]
    W -->|"7 d, not 30 d"| EME["emerging<br/>newly commissioned → act now"]
    W -->|"24 h only"| ACU["acute<br/>a fire → send a truck, not a notice"]
    CHR --> Z["Cluster cells into ZONES<br/>an inspector visits a zone, not a hexagon"]
    EME --> Z
    ACU --> Z
    Z --> A{"Does any instrument<br/>point at a PLACE?"}
    A -->|"OSM site within 3 km, or fire, or SO₂/aerosol tracer"| ENF["ENFORCEABLE<br/>→ attribution + action queue"]
    A -->|"only high NO₂ over dense roads"| DIF["DIFFUSE urban background<br/>real pollution, nobody to serve a notice on<br/>→ policy target, stays on the map"]
```

Three rules hold this together:

- **Never the mean.** Every aggregate is a **median**, every spread a **MAD**. In
  this domain the outliers *are* the phenomenon: a mean lets one spike hour
  manufacture a chronic source out of a single bonfire.
- **One window is not a signal.** A real-time spike is noise. A source is what is
  still there when you zoom out.
- **Contrast, not rank.** Compare a cell to *its own neighbourhood*, not to the
  city. "This district is dense" is true, unactionable, and not a violator.

---

## The citizen loop — from a phone to the evidence chain

Citizens are a third observation tier — *"900 stations + 2 satellites + a million
human sensors"* — and, uniquely, the only instrument that can see the construction
dust the satellite is blind to. The intake is a live Telegram bot (and a web form
against the same endpoint), deployed on n8n behind automatic HTTPS.

<p align="center">
  <img src="./docs/img/citizen-find-ward.png" alt="AirCase citizen entry: citywide AQI, search by ward name, use my location, or tap the map" width="34%" />
</p>

<p align="center">
  <sub>A citizen never types a ward ID. Search by name, share location, or tap the
  hexagon they are standing in — and the citywide number is there before they
  pick, so the page is useful on the first screen.</sub>
</p>

```mermaid
flowchart LR
    U["Citizen<br/><small>“Bhalswa mein kachra<br/>jal raha hai”</small>"] --> TG["Telegram / web form"]
    TG --> N8N

    subgraph N8N["n8n workflow — channels stay DUMB"]
        direction TB
        RT["Preprocess · route<br/><small>command? inspector? report?</small>"] --> GEM["Gemini · extract<br/><small>{category, ward_guess}</small>"]
        GEM --> FIN["Finalize · DECIDE<br/><small>canonicalise ward vs 733 official names;<br/>clamp category; keyword fallback</small>"]
    end

    FIN --> SUP["Supabase<br/><small>citizen_reports (RLS: anon insert-only)</small>"]
    SUP --> SYNC["sync_supabase.py<br/><small>→ data/outputs/</small>"]
    SYNC --> ATT["attribute<br/><small>matches by ward + evidence window</small>"]
    ATT --> EV["Evidence chain<br/><small>“1 citizen report of waste_burning”<br/>confidence 0.77 → 0.84</small>"]

    classDef llm fill:#a9760a22,stroke:#a9760a;
    classDef det fill:#0f6b3f22,stroke:#0f6b3f;
    class GEM llm
    class FIN,SYNC,ATT det
```

**Proven end-to-end**, not just wired. A live report against a detected
`waste_burning` zone lifts its confidence **0.77 → 0.84** and adds
*"Citizen reports of waste burning"* to its evidence factors — because it counts as
**one more independent agreeing instrument** (fire + citizen = 2), computed by
deterministic math. The lift is **capped**: a brigade of reports can never
manufacture a source or out-shout the satellite. Reports outside the evidence
window, or that match no real ward, correctly do **not** corroborate.

> **Honest footnote.** That lift was demonstrated live, and it is **not present in
> the shipped Delhi data** — 0 of 53 attributions carry citizen corroboration. The
> reports are from today; the Delhi case study is a November 2025 window. Citizens
> cannot report into a historical window, and we will not widen the evidence window
> to manufacture a match. It is a live capability shown on a historical case.

The same bot carries the **inspector loop** — an inspector replies `done <id>`, the
status is written back, and the ledger stamps the response time. Two audiences, one
channel; the channel moves bytes and nothing more.

---

## What's real vs prototype

Because a README that oversells is the same bug as a metric that oversells.

| | Status |
|---|---|
| Ingestion — Sentinel-5P, OpenAQ, Open-Meteo, FIRMS, OSM | ✅ **all real and live.** Run end-to-end on Delhi, Nov 2025 |
| H3 spatial fabric + real ward boundaries | ✅ real — **official Datameet GeoJSON for all eight cities**, 1,004 municipal wards over 7,931 H3 res-8 cells. No Voronoi fallback is in use anywhere |
| Detection (NO₂ contrast + fire persistence) | ✅ real — validated on a real landfill fire |
| Attribution + evidence chain + confidence | ✅ real, truth-scored, calibrated |
| Forecast — 3-hourly to +72 h vs persistence baseline | ✅ real. One pooled LightGBM with the lead time as a feature, scored at **24 lead times** on a held-out tail. See [the sawtooth](#the-evaluation-was-flattering-our-own-baseline) — and the city where we lose |
| Prioritisation (EPS) + dispatch routing | ✅ real — deterministic score, greedy set-cover, per-team routes |
| Enforcement memo + rule-matched legal basis | ✅ real — deterministic legal citation, LLM drafts prose only |
| Ward advisory agent — English + city language | ✅ real — **every ward, not a sample**: Delhi 267 wards in en/hi, Chennai 177 in en/ta, Bengaluru 227 in en/kn, with per-language verification labels (native-speaker vs cross-checked — never claims a review it didn't get) |
| Intervention ledger — response time + counterfactual | ✅ real. Response-time is honest (CAG's *weeks* vs one automated batch); effectiveness freezes the +48 h counterfactual, `our_impact: null` until a real intervention exists |
| Channels — n8n citizen intake + inspector loop | ✅ **live.** Telegram bot + web webhook → Supabase, proven end-to-end into the evidence chain |
| Jurisdiction tagging on the action queue | ✅ real — the H3 fabric is a rectangular bbox but a municipality is an irregular polygon inside it, so **165 of 580 hotspot cells and 12 of 46 actions fall outside city limits** (all of Pune's). Tagged *"refer to state board"*, not dropped: a landfill past the boundary still pollutes the city, so deleting it discards a true detection — the routing changes, the finding stands |
| Read-only serving API (24 endpoints) | ✅ real — one instance serves all eight cities, co-located with the channel layer behind one certificate |
| **Fusion exposure field** | ❌ **claim withdrawn, now on EIGHT cities.** Leave-one-station-out, it loses to a naive city-mean on **7 of 8**; two have a *negative* R² (Pune −0.244, Hyderabad −0.221) — worse than predicting the mean. The single win is Delhi (+2.1%), the city with 24 stations, the most we have anywhere. We tried predicting the *deviation* from the city median — a construction that cannot lose to the baseline, since a zero residual **is** the baseline — and it still lost, which means the model fits its training stations' siting quirks rather than structure that transfers. With ~24 stations we cannot demonstrate spatial skill, so we do not claim it. **Detection is the contribution.** |
| **Frontend — Next.js + deck.gl console** | ✅ **deployed.** Admin console and citizen view live on Vercel, reading the live API with a static-bundle fallback so the map still renders if the backend is down |
| GEE Sentinel-5P collector | ✅ **built and wired** — real `COPERNICUS/S5P` extraction; it produced the real Delhi result. Live satellite needs GEE auth on the run machine (`gcloud auth application-default login`); until then synthetic mode runs fully offline |
| **Intervention effectiveness** | ❌ **not claimed.** The ledger freezes the +48 h counterfactual at dispatch and waits for a real actioned outcome. Zero of four actions are actioned, so `our_impact` is `null`. Response *time* is real; response *effect* is not yet measurable |

<p align="center">
  <img src="./docs/img/ledger.png" alt="AirCase intervention ledger: response time marked measured, effectiveness marked not yet measured, with the reason stated" width="62%" />
</p>

<p align="center">
  <sub>The ledger keeps those two apart on purpose. <b>Response time: measured.</b>
  <b>Effectiveness: not yet measured</b> — "nobody has acted on these yet, so we do
  not claim an impact number; attributing natural change to ourselves would be
  dishonest." A product that reported an impact here would be easy to build and
  impossible to defend.</sub>
</p>
| Voice advisory **audio** (TTS) | ✅ real for **three** cities — Google TTS, **50 clips over the 25 highest-risk wards** of Delhi (hi), Chennai (ta) and Bengaluru (kn), played in the citizen view. The cap is deliberate: 227 wards × 2 languages is ~450 sequential TTS calls, and an advisory reading "air quality is Good" does not need a voice note. Every ward still gets the **text**. The other five cities have none — the agent needs Google Cloud TTS and that project's billing is gone |
| Network audit — monitoring blind spots | ✅ real — **24 of Delhi's 1,703 cells are monitored (1.4%)**; the 40 worst blind spots are ranked into a next-sensor placement list |
| Multi-city — one instance, **eight** cities | ✅ real — `?city=` on every endpoint. Delhi, Chennai, Bengaluru, Mumbai, Kolkata, Hyderabad, Pune and Ahmedabad each have a full live pipeline run over the same 90-day window, on real municipal ward boundaries |

> ### ⚠️ Live mode refuses to fake anything
> If the satellite, fire, or OSM collector fails, the pipeline **raises rather than
> substituting synthetic data**. Each of those layers *invents a place we would then
> accuse* — a fabricated output, not a degraded one. (Stations may degrade: they
> feed the exposure map, not the detector.)
>
> ### ⚠️ Don't run it in monsoon
> July is the worst possible month for both instruments: cloud masks 71% of the NO₂
> retrieval, and **nothing burns when it is wet** — real FIRMS returns **2 fires over
> Bengaluru in 60 days** (our synthetic world had 281). Both channels go blind.
> Delhi's burning season is **October–November**, which is where the real result
> above comes from.

---

## See it running

It is deployed. These are live, not screenshots.

| | |
|---|---|
| **Admin console** | [aircase-aq.vercel.app/admin](https://aircase-aq.vercel.app/admin) — map, hotspot zones, evidence chains, EPS queue, dispatch routes, agent runner |
| **Citizen view** | [aircase-aq.vercel.app/citizen](https://aircase-aq.vercel.app/citizen) — your ward's AQI, *why* it is bad, what is being done, and a 3-hourly timeline |
| **API** | [/docs](https://aq-intel.duckdns.org/aircase/docs) — 24 endpoints, `?city=` on each, OpenAPI schema |
| **Health** | [/health](https://aq-intel.duckdns.org/aircase/health) — which cities this instance can answer for |
| **Telegram** | [@aircaseaqbot](https://t.me/aircaseaqbot) — send a photo, a voice note, or a sentence in your own language |

The console runs the agent chain live: pick an agent (or all nine), set the
inspection-team count, and watch dispatch recluster. Everything read-only falls
back to a committed static bundle, so the map still renders if the backend is
down.

<p align="center">
  <img src="./docs/img/chennai-zones.png" alt="AirCase console switched to Chennai: twelve enforcement zones in the queue, a different coastline, the same pipeline" width="100%" />
</p>

<p align="center">
  <sub><b>Same code, different city.</b> Chennai: twelve enforcement zones, on a
  satellite feed only 26% complete and four CPCB stations for eleven million
  people. No new hardware, no city-specific code — a bounding box and a ward
  layer. That is how this reached <b>eight cities</b>: the marginal cost of the
  ninth is compute, not sensors.</sub>
</p>

The Telegram loop closes in both directions, in the citizen's own language and
in their own medium:

<p align="center">
  <img src="./docs/img/telegram-voice-loop.png" alt="Telegram: a voice note in Hinglish is transcribed, logged as ward DELHI CANTT CHARGE 1 category waste_burning, and answered with a Hindi audio advisory" width="62%" />
</p>

A **spoken** report — *"Delhi Cant, Charj 1 mein kachra jal raha hai"* — is
transcribed, the ward is canonicalised against the official list (`Charj 1` →
`DELHI CANTT CHARGE 1`), the category is clamped to `waste_burning`, and the
ward's advisory comes back **as audio**. No typing, no app, no literacy
requirement. The people most exposed to this air are outdoor workers.

---

## The 100% trap

This project once reported **100% attribution accuracy**. It was an artefact, and
the story is worth the two minutes.

The synthetic world was emitting its hidden sources straight into the OSM layer
with **exact coordinates and exact category labels**, and dispersing them with the
*same* `exp(-d/2)·wind_alignment` kernel the attribution scorer uses. The scorer
was being handed the answer key and congratulated for reading it.

`ingestion/synthetic.py` is now deliberately **adversarial**:

- **different physics** — a Gaussian plume the scorer does not assume
- **sources that appear on no map at all** (illegal burning files no paperwork)
- **decoy sites** that are on the map and emit nothing
- **a satellite blurred to its true ~5.5 km footprint**
- **column-vs-surface decoupling** — the satellite sees no boundary-layer trapping;
  a station does. Bridging that gap is the fusion model's actual job

The score collapsed. *That collapse was the finding.* Every number in this README
survived the rebuild.

We caught the same class of error a second time: the stat *"0 of 9 sources sit
within 2 km of a monitor"* was **guaranteed by the world model's own placement
rule** (~99% true by construction) and was being reported as a discovery. It has
been replaced by the unbiased-network number above, which owes nothing to any
assumption we made.

> **If a number ever comes back at 100%, assume leakage before you assume success.**

---

## Repo layout

```
app/            backend/main.py — read-only FastAPI (24 endpoints, ?city= scoped)
                frontend/       — Next.js + deck.gl console + citizen view
ingestion/      collectors (6 sources, live + synthetic fallback)
                preprocessing/panel.py — the cell × hour feature table
                synthetic.py — the adversarial hidden-source world
intelligence/   orchestrator.py     — the 9-agent LangGraph state machine
                models/fusion.py    — LightGBM exposure field
                models/signals.py   — robust multi-window statistics
                agents/             — detect, attribution, forecast, prioritise,
                                      memo, advisory, voice, ledger, audit
                                      + llm_gateway
shared/         config, H3 grid utilities, real ward layer (8 cities)
scripts/        run_pipeline.py, sync_supabase.py + truth-scored evaluations
db/             schema.sql — Supabase contracts the channel layer writes to
deploy/         n8n on GCP: Terraform + Caddy + DuckDNS runbook
docs/           architecture.md — the 9-layer design and why it's shaped this way
```

## Evaluations

Every claim in this README is a script you can run.

```bash
python scripts/eval_detection.py            # sources found vs missed; enforceable-zone precision
python scripts/eval_attribution.py          # accuracy, split registered vs unregistered; confidence calibration
python scripts/eval_station_sensitivity.py  # is the headline an artefact of where we put the monitors? (no)
python scripts/eval_hotspot_recovery.py     # fusion as an EXPOSURE map — and why it is not a detector
```

## Roadmap

**Shipped.** Sentinel-5P via Google Earth Engine · 9-agent LangGraph pipeline ·
real Datameet ward boundaries for **all eight cities** (1,004 wards, 7,931 H3
cells, 17.1 M cell-hours) · n8n citizen intake, Telegram bot and inspector loop,
proven end-to-end into the evidence chain · multi-language advisory text and
voice · Next.js console and citizen view on Vercel · read-only API and live agent
runs on AWS behind Caddy, serving **all eight cities from one instance**, sharing
one box and one certificate with the channel layer.

**Known gaps, stated rather than hidden:**

- [ ] **Construction dust is invisible to us** — recall 0/3. It is coarse PM with
      no satellite tracer and it does not burn. Closing it needs Sentinel-2 optical
      change detection, the OSM permit layer, and citizen reports at volume.
- [ ] **Traffic corridors are confounded** — recall 0/4. The road network lifts NO₂
      across the whole core, so a corridor cannot out-shout its own neighbourhood.
- [ ] **Citizen corroboration has never fired on the shipped data.** The path works
      and is proven — a Bhalswa report resolves to the right ward and lifts
      confidence — but the reports are from today and the Delhi case study is
      November 2025. Citizens cannot report into a historical window.
- [ ] **Intervention effectiveness is unmeasured.** The counterfactual is frozen at
      dispatch; it needs a real actioned inspection to become an impact number.
- [ ] **No rate limiting on the public report webhook.** It is an unauthenticated
      intake form by design; abuse handling is not built.
- [ ] **26% of enforcement actions fall outside municipal jurisdiction.** The H3
      fabric is a rectangular bbox; a municipality is an irregular polygon inside
      it, so 12 of 46 actions (and *all* of Pune's) sit beyond the city boundary.
      They are now tagged *"refer to state board"* rather than dropped — the
      detection stands, the routing changes — but the enforcement queue should
      filter by jurisdiction upstream, in `prioritise.py`, not in the console.
- [ ] **A live LLM makes the agent chain scale with hotspot count.** Attribution
      calls the model once per hotspot, so Delhi's 53 take ~6.5 min against
      Ahmedabad's 27 at 114 s. Fine for a batch job, too slow for a request a
      human is watching; it wants batching or streaming per-hotspot results.
- [ ] Kannada advisory text is `cross_checked`, not native-verified.

## Contributors

<p align="center">
  <img alt="shyam" src="https://img.shields.io/badge/Shyam-Backend%20%2B%20Frontend-0f6b3f?style=flat-square" />
  <img alt="keshav" src="https://img.shields.io/badge/Keshav-Backend%20%2B%20Frontend%20%2B%20Agents-0f6b3f?style=flat-square" />
  <img alt="suyash" src="https://img.shields.io/badge/Suyash-EPS%20%2B%20Deployment-a9760a?style=flat-square" />
  <img alt="saumya" src="https://img.shields.io/badge/Saumya%20Saraswat-Intelligence%20%2B%20Frontend%20Inputs-a9760a?style=flat-square" />
</p>

| Contributor | Focus areas |
|---|---|
| **Saumya Saraswat** | Intelligence, with inputs on Frontend |
| **Suyash Mittal** | EPS, Deployment |
| **Keshav Agrawal** | Backend, Frontend, Agents |
| **Shyamsundar Paramasivam** | Backend, Frontend |


---

<p align="center">
  <sub>Built for a hackathon. The hardest engineering here wasn't the models — it was
  building an evaluation honest enough to tell us the models were wrong.</sub>
</p>
