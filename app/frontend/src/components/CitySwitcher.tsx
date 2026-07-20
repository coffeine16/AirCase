"use client";
/**
 * CitySwitcher — the multi-city capability, made obvious. Switching moves the
 * whole app: every city-scoped hook re-reads from the new city's static bundle.
 */
import { useState, useRef, useEffect } from "react";
import { useCity, CITIES } from "@/lib/CityContext";

export default function CitySwitcher() {
  const { city, cityLabel, setCity } = useCity();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="btn btn-ghost btn-sm"
        style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600 }}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent-emerald)", display: "inline-block" }} />
        {cityLabel}
        <span style={{ fontSize: "0.6rem", opacity: 0.7, transform: open ? "rotate(180deg)" : "none", transition: "transform var(--transition-fast)" }}>▾</span>
      </button>

      {open && (
        <div
          role="listbox"
          className="card"
          style={{
            position: "absolute", top: "calc(100% + 6px)", right: 0, zIndex: "var(--z-modal)",
            minWidth: 160, padding: 4, boxShadow: "var(--shadow-lg)",
          }}
        >
          {CITIES.map((c) => (
            <button
              key={c.id}
              role="option"
              aria-selected={c.id === city}
              onClick={() => { setCity(c.id); setOpen(false); }}
              style={{
                display: "flex", alignItems: "center", gap: 8, width: "100%",
                padding: "8px 10px", borderRadius: "var(--radius-sm)", cursor: "pointer",
                background: c.id === city ? "var(--bg-active)" : "transparent",
                color: c.id === city ? "var(--accent-blue)" : "var(--text-secondary)",
                fontSize: "0.85rem", fontWeight: c.id === city ? 600 : 400,
                border: "none", textAlign: "left",
                transition: "background var(--transition-fast)",
              }}
              onMouseEnter={(e) => { if (c.id !== city) e.currentTarget.style.background = "var(--bg-hover)"; }}
              onMouseLeave={(e) => { if (c.id !== city) e.currentTarget.style.background = "transparent"; }}
            >
              {c.id === city && <span style={{ color: "var(--accent-emerald)" }}>✓</span>}
              <span style={{ marginLeft: c.id === city ? 0 : 16 }}>{c.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
