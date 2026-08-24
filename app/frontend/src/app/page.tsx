import Link from "next/link";
import type { Metadata } from "next";
import { icon, MapIcon, Users, ArrowRight, Hexagon } from "@/components/Icon";

export const metadata: Metadata = {
  title: "AirCase — Select Role",
  description: "AI-powered urban air quality intelligence: signal → attribution → action.",
};

/**
 * Landing. Rebuilt from a five-colour emoji grid to a single-accent, typographic
 * page. What changed and why:
 *  · The gradient-filled wordmark is gone. Gradient text is the most-copied
 *    "startup landing" tell there is, and it fought the near-neutral chrome the
 *    rest of the app now uses.
 *  · The stat row used a DIFFERENT accent per tile (emerald/blue/purple/amber).
 *    Colour there carried no information — the tiles are not categories — so it
 *    was decoration that made five equal facts look like five warnings. They are
 *    now one hairline rule and one type scale.
 *  · Emoji tiles (🗺️ 🏙️ 🌏 🛰️ 🤖) → Lucide glyphs that inherit text colour.
 */

const STATS = [
  { value: "8",    label: "Real cities",  sub: "live, not synthetic" },
  { value: "6",    label: "Data sources", sub: "satellite to citizen" },
  { value: "9",    label: "AI agents",    sub: "detect → dispatch" },
  { value: "4",    label: "Languages",    sub: "EN · HI · TA · KN" },
  { value: "460m", label: "Grid",         sub: "H3 res-8 hexes" },
];

const ROLES = [
  {
    href: "/admin",
    icon: MapIcon,
    title: "Admin Console",
    body: "Interactive map, agent control, the enforcement queue, and the evidence chain behind every accusation.",
    cta: "Open console",
  },
  {
    href: "/citizen",
    icon: Users,
    title: "Citizen View",
    body: "Your ward's air quality, a 72-hour forecast, a multilingual advisory, and pollution reporting.",
    cta: "Open citizen view",
  },
];

export default function LandingPage() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "var(--space-2xl) var(--space-lg)",
        gap: "var(--space-2xl)",
      }}
    >
      <div style={{ textAlign: "center", maxWidth: 620 }}>
        <div
          style={{
            display: "inline-flex", alignItems: "center", gap: 7,
            padding: "4px 10px", marginBottom: "var(--space-lg)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-full)",
            fontSize: "0.7rem", fontWeight: 520,
            color: "var(--text-tertiary)",
          }}
        >
          <span
            aria-hidden
            className="animate-breathe"
            style={{
              width: 5, height: 5, borderRadius: "50%",
              background: "var(--positive)", display: "inline-block",
            }}
          />
          8 Indian cities — live
        </div>

        <h1 className="display" style={{ marginBottom: "var(--space-md)" }}>
          AirCase
        </h1>

        <p className="lede">
          From AQI dashboards to enforcement dispatch. Names <strong>who</strong> is
          polluting, <strong>where</strong>, with <strong>what evidence</strong>, and{" "}
          <strong>what to do about it today</strong>.
        </p>
      </div>

      {/* Role selection — the page's only two actions, given equal weight
          because the audiences are genuinely separate. */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "var(--space-md)",
          width: "100%",
          maxWidth: 680,
        }}
      >
        {ROLES.map(({ href, icon: Glyph, title, body, cta }) => (
          <Link key={href} href={href} style={{ textDecoration: "none" }}>
            <div
              className="card card-hover"
              style={{
                padding: "var(--space-lg)",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <Glyph {...icon.md} style={{ color: "var(--accent)" }} aria-hidden />
              <h3 style={{ color: "var(--text-primary)" }}>{title}</h3>
              <p style={{ fontSize: "0.83rem", flex: 1 }}>{body}</p>
              <span
                style={{
                  display: "inline-flex", alignItems: "center", gap: 5,
                  color: "var(--accent)", fontSize: "0.8125rem", fontWeight: 550,
                }}
              >
                {cta}
                <ArrowRight {...icon.md} aria-hidden />
              </span>
            </div>
          </Link>
        ))}
      </div>

      {/* Facts, not badges. One rule, one scale, no colour. */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(118px, 1fr))",
          gap: 0,
          width: "100%",
          maxWidth: 680,
          borderTop: "1px solid var(--border-subtle)",
        }}
      >
        {STATS.map((s) => (
          <div key={s.label} style={{ padding: "var(--space-md) var(--space-sm)" }}>
            <div
              className="mono"
              style={{
                fontSize: "1.3rem", fontWeight: 600, lineHeight: 1.1,
                letterSpacing: "-0.02em", color: "var(--text-primary)",
              }}
            >
              {s.value}
            </div>
            <div style={{ fontSize: "0.76rem", fontWeight: 520, color: "var(--text-secondary)", marginTop: 3 }}>
              {s.label}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)" }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div
        style={{
          display: "inline-flex", alignItems: "center", gap: 6,
          fontSize: "0.7rem", color: "var(--text-tertiary)",
        }}
      >
        <Hexagon {...icon.sm} aria-hidden />
        H3 resolution 8 · deterministic scoring · LLMs explain, never rank
      </div>
    </main>
  );
}
