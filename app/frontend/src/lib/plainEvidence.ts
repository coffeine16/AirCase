/**
 * plainEvidence — turn the attribution agent's evidence strings into sentences
 * a resident can actually read.
 *
 * WHY THIS EXISTS, AND WHY IT IS FRONTEND-ONLY
 * The agent writes evidence for an INSPECTOR: "waste_burning_1181342843 is 1.53
 * km away, wind alignment 0.47" is exactly right for someone building a case
 * file — it is precise, it carries the OSM id you would look up, and the cosine
 * is the number the score was computed from. Rendered to a member of the public
 * it is close to meaningless, and it leaks a database key onto a public page.
 *
 * The admin console keeps the raw strings deliberately. Only the citizen
 * surface is rewritten, because the two audiences need different things from
 * the same evidence. Nothing here changes what was concluded — this is a
 * translation layer, not a second opinion.
 *
 * THE FOUR RULES
 *   1. Never print an OSM id. An unnamed site is described by what it is.
 *   2. Never print a bare cosine. Wind alignment becomes words.
 *   3. Never print jargon unexplained. "Boundary layer" says what it means.
 *   4. Drop the model talking about itself. "Highest deterministic category
 *      score (0.68)" tells a citizen nothing about their air.
 */

/** Raw OSM-derived ids the pipeline emits when a site has no name in OSM. */
const OSM_ID = /\b(waste_burning|industrial|construction|traffic|road)_\d+\b/gi;

const KIND_NOUN: Record<string, string> = {
  waste_burning: "a waste-burning site",
  industrial: "an industrial site",
  construction: "a construction site",
  traffic: "a major road",
  road: "a major road",
};

/** Self-referential scoring lines. These describe the ranking arithmetic, not
 *  the world, and are dropped rather than translated. */
const SELF_REFERENTIAL =
  /\b(deterministic|category)\s+score\b|\bscore\s+of\s+[\d.]+|\bhighest\s+\w+\s+score\b/i;

/** Channels we do not stand behind (measured as instrument noise — see
 *  detect.py::NOISE_CHANNELS). Kept out of the public evidence chain. */
const NOISE_CHANNEL = /\b(aai|aerosol|so2|so₂)\b/i;

/** A cosine in [0,1] between wind direction and the bearing from a candidate
 *  source. Nobody outside the project reads that as a number. */
function windPhrase(alignment: number): string {
  if (!Number.isFinite(alignment)) return "nearby";
  if (alignment >= 0.7) return "and the wind blows almost straight from it towards your ward";
  if (alignment >= 0.4) return "and the wind often blows from it towards your ward";
  if (alignment >= 0.15) return "and the wind sometimes blows from it towards your ward";
  return "though the wind only rarely carries its smoke your way";
}

/** Same scale as windPhrase, in the gerund, for use after "with the wind …".
 *  Written out rather than derived from windPhrase by string surgery — that
 *  produced "with the wind often blows from it". */
function windPhraseGerund(alignment: number, subject: string): string {
  if (!Number.isFinite(alignment)) return `blowing near ${subject}`;
  if (alignment >= 0.7) return `blowing almost straight from ${subject} towards your ward`;
  if (alignment >= 0.4) return `often blowing from ${subject} towards your ward`;
  if (alignment >= 0.15) return `sometimes blowing from ${subject} towards your ward`;
  return `only rarely carrying their smoke your way`;
}

/** "waste_burning_118…" -> "a waste-burning site". A real OSM name is kept. */
function siteLabel(raw: string): string {
  const name = raw.trim().replace(/^the\s+/i, "");
  const m = name.match(/^(waste_burning|industrial|construction|traffic|road)_\d+$/i);
  if (m) return KIND_NOUN[m[1].toLowerCase()] ?? "a nearby site";
  return name;
}

function km(v: number): string {
  return v < 1 ? `${Math.round(v * 1000)} m` : `${v.toFixed(1)} km`;
}

/**
 * Translate one evidence factor. Returns null when the line should not be
 * shown to a citizen at all.
 */
export function plainEvidence(factor: string): string | null {
  const f = factor.trim();
  if (!f) return null;
  if (SELF_REFERENTIAL.test(f)) return null;
  if (NOISE_CHANNEL.test(f)) return null;

  // ── Trapped air ─────────────────────────────────────────────────────────────
  // "shallow boundary layer (100 m) trapping emissions"
  // "Trapped air conditions with a low boundary layer height of 105 m"
  // "Meteorological conditions with air trapped and low BLH (120 m)"
  const blh = f.match(/(?:boundary layer|blh)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*m/i);
  if (blh) {
    const h = Math.round(parseFloat(blh[1]));
    return `The air was sitting trapped about ${h} m above the ground, so smoke could not blow away and built up instead.`;
  }

  // ── Fires ───────────────────────────────────────────────────────────────────
  // "satellite fire detections in 30 hours (18% of the window)"
  const fireWin = f.match(/fire detections? in (\d+) hours?\s*\((\d+(?:\.\d+)?)%/i);
  if (fireWin) {
    return `Satellites spotted fires burning near you during ${fireWin[1]} separate hours — about ${Math.round(
      parseFloat(fireWin[2]),
    )}% of the period we looked at.`;
  }
  // "22 fire hours recorded in fire activity data" / "6 fire hours within the evidence window"
  const fireHrs = f.match(/(\d+)\s*fire hours?/i);
  if (fireHrs) {
    return `Satellites spotted fires burning near you during ${fireHrs[1]} separate hours.`;
  }

  // ── NO2 ─────────────────────────────────────────────────────────────────────
  // "NO2 column at p97 citywide" / "High NO2 column density at the 93rd city percentile"
  const no2 = f.match(/no.?2[^0-9]*(?:p|at the |at )?(\d{1,2})(?:st|nd|rd|th)?\s*(?:city )?percentile|no.?2 column at p(\d{1,2})/i);
  if (/no.?2/i.test(f) && no2) {
    const pct = no2[1] ?? no2[2];
    return `Nitrogen dioxide — the gas given off by burning — was higher here than in ${pct}% of the city.`;
  }

  // ── Land use ────────────────────────────────────────────────────────────────
  // "Landuse context includes 5 construction features"
  const lu = f.match(/land.?use context includes (\d+)\s+(\w+)/i);
  if (lu) {
    const n = lu[1];
    const kind = lu[2].toLowerCase().replace(/_/g, " ");
    return `There ${n === "1" ? "is" : "are"} ${n} ${kind} site${n === "1" ? "" : "s"} mapped close by.`;
  }
  // "Presence of 4 industrial land-use features and multiple upwind industrial
  //  candidates with wind alignment up to 0.8"
  const luPresence = f.match(/presence of (\d+)\s+(\w[\w-]*)\s+land.?use/i);
  if (luPresence) {
    const n = luPresence[1];
    const kind = luPresence[2].toLowerCase().replace(/-/g, " ");
    const upwind = /upwind/i.test(f) ? ", several of them upwind of you" : "";
    return `There ${n === "1" ? "is" : "are"} ${n} ${kind} site${n === "1" ? "" : "s"} mapped close by${upwind}.`;
  }

  // ── A named candidate, its distance, and the wind ───────────────────────────
  // "X is 1.53 km away, wind alignment 0.47"
  // "Municipal Landfill located 0.44 km away with 0.61 wind alignment"
  let cand = f.match(/^(.*?)\s+is\s+(\d+(?:\.\d+)?)\s*km away,\s*wind alignment\s*(\d*\.?\d+)/i);
  if (!cand) {
    const alt = f.match(/^(.*?)\s+located\s+(\d+(?:\.\d+)?)\s*km away with\s+(\d*\.?\d+)\s*wind alignment/i);
    if (alt) cand = alt;
  }
  if (cand) {
    const label = siteLabel(cand[1]);
    const dist = km(parseFloat(cand[2]));
    return `${capitalise(label)} is ${dist} away, ${windPhrase(parseFloat(cand[3]))}.`;
  }

  // "Close proximity of A (0.9 km) and B (1.4 km, 0.62 wind alignment)"
  const twin = f.match(
    /close proximity of\s+(.+?)\s*\((\d+(?:\.\d+)?)\s*km\)\s*and\s+(.+?)\s*\((\d+(?:\.\d+)?)\s*km(?:,\s*(\d*\.?\d+)\s*wind alignment)?\)/i,
  );
  if (twin) {
    const a = siteLabel(twin[1]);
    const b = siteLabel(twin[3]);
    const dA = km(parseFloat(twin[2]));
    const dB = km(parseFloat(twin[4]));
    // Both unnamed sites of the same kind collapse to the same label, so
    // "a construction site and a construction site" has to become "two of them".
    const pair =
      a === b
        ? `Two ${plural(a)} sit close by — one ${dA} away and another ${dB} away`
        : `${capitalise(a)} sits ${dA} away and ${b} ${dB} away`;
    const wind = twin[5]
      ? `, with the wind ${windPhraseGerund(parseFloat(twin[5]), a === b ? "them" : "one of them")}`
      : "";
    return `${pair}${wind}.`;
  }

  // "Highest wind alignment (0.81) for waste_burning_118… candidate"
  const bestWind = f.match(/highest wind alignment\s*\((\d*\.?\d+)\)\s*for\s+(.+?)\s*candidate/i);
  if (bestWind) {
    return `Of everything nearby, ${siteLabel(bestWind[2])} is what the wind most often carries towards your ward.`;
  }

  // ── Fallback ────────────────────────────────────────────────────────────────
  // An unrecognised phrasing (the pipeline gains new ones over time) is shown
  // rather than dropped — losing evidence is worse than showing it plainly —
  // but never with a raw id in it.
  const safe = f.replace(OSM_ID, (m) => {
    const kind = m.split("_").slice(0, -1).join("_").toLowerCase();
    return KIND_NOUN[kind] ?? "a nearby site";
  });
  return capitalise(safe);
}

/** "a construction site" -> "construction sites" (for the two-candidate case). */
function plural(label: string): string {
  return label.replace(/^(a|an)\s+/i, "") + "s";
}

function capitalise(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Translate a whole evidence chain, dropping what a citizen should not see and
 *  de-duplicating lines that collapse to the same sentence. */
export function plainEvidenceChain(factors: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const f of factors) {
    const p = plainEvidence(f);
    if (!p || seen.has(p)) continue;
    seen.add(p);
    out.push(p);
  }
  return out;
}
