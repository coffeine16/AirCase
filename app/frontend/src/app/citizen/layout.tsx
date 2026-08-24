import type { Metadata } from "next";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import CitySwitcher from "@/components/CitySwitcher";
import { icon, Wind } from "@/components/Icon";

export const metadata: Metadata = {
  title: "Citizen View — AirCase",
  description: "Your ward's air quality, 72-hour forecast, health advisory, and pollution reports.",
};

export default function CitizenLayout({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-base)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-md)",
          padding: "0 var(--space-md)",
          height: 52,
          background: "var(--bg-primary)",
          borderBottom: "1px solid var(--border-subtle)",
          position: "sticky",
          top: 0,
          zIndex: "var(--z-header)",
        }}
      >
        <Link
          href="/citizen"
          style={{
            display: "flex", alignItems: "center", gap: 9,
            textDecoration: "none", color: "var(--text-primary)", minWidth: 0,
          }}
        >
          {/* Was a 🌿 emoji in a green-tinted rounded box — a leaf, for an air
              QUALITY warning service. The glyph now matches the subject and
              takes the accent like everything else. */}
          <Wind {...icon.md} aria-hidden style={{ color: "var(--accent)", flexShrink: 0 }} />
          <div className="hide-mobile" style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 600, fontSize: "0.875rem", lineHeight: 1.25, whiteSpace: "nowrap" }}>
              AirCase
            </div>
            {/* City now lives only in CitySwitcher's own label (nav, right side)
                — showing it here too was the same fact twice in one header. */}
            <div style={{ fontSize: "0.7rem", color: "var(--text-tertiary)", lineHeight: 1.2, whiteSpace: "nowrap" }}>
              Your ward air quality
            </div>
          </div>
        </Link>

        <nav style={{ display: "flex", gap: 2, alignItems: "center" }}>
          <Link href="/citizen/explore" className="nav-link">Explore</Link>
          <Link href="/citizen/reports" className="nav-link hide-mobile">My reports</Link>
          <Link href="/admin" className="nav-link">Admin</Link>
          <CitySwitcher />
          <ThemeToggle />
        </nav>
      </header>

      <main style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {children}
      </main>
    </div>
  );
}
