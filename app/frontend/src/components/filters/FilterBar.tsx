"use client";
import { useMemo, useState } from "react";
import type { MapFilters, HotspotKind, SourceCategory, Ward } from "@/lib/types";
import { PERSISTENCE_LABELS, SOURCE_LABELS } from "@/lib/constants";
import { icon, Search, X } from "@/components/Icon";

interface Props {
  filters: MapFilters;
  wardCells?: Ward[];
  onWards: (ids: string[]) => void;
  onSources: (s: SourceCategory[]) => void;
  onPersistence: (p: HotspotKind[]) => void;
  /** Fly the map to this ward, independent of filtering — searching for a
   *  ward should locate it even if it has nothing in it to filter. */
  onFocusWard?: (wardId: string) => void;
  onReset: () => void;
}

const PERSISTENCE_OPTIONS: HotspotKind[] = ["chronic", "emerging", "acute"];
const SOURCE_OPTIONS: SourceCategory[] = ["industrial", "waste_burning", "construction", "traffic"];

// The chip's active colour is the DATA colour of the thing it filters, passed
// through --tint. That is the one place colour is allowed to be loud here,
// because it matches what the map draws.
const PERSIST_TINT: Record<HotspotKind, string> = {
  chronic:  "var(--persist-chronic)",
  emerging: "var(--persist-emerging)",
  acute:    "var(--persist-acute)",
};
const SOURCE_TINT: Record<SourceCategory, string> = {
  industrial:    "var(--source-industrial)",
  waste_burning: "var(--source-waste-burning)",
  construction:  "var(--source-construction)",
  traffic:       "var(--source-traffic)",
};

function ChipGroup<T extends string>({
  label, options, selected, onToggle, labels, tints,
}: {
  label: string;
  options: T[];
  selected: T[];
  onToggle: (val: T) => void;
  labels: Record<T, string>;
  tints?: Partial<Record<T, string>>;
}) {
  return (
    <fieldset style={{ border: "none" }}>
      <legend className="section-label" style={{ marginBottom: 6 }}>{label}</legend>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {options.map((opt) => {
          const active = selected.includes(opt);
          return (
            <button
              key={opt}
              type="button"
              className="chip"
              data-active={active}
              aria-pressed={active}
              onClick={() => onToggle(opt)}
              style={tints?.[opt] ? ({ ["--tint" as string]: tints[opt] }) : undefined}
            >
              {labels[opt]}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

function toggle<T>(arr: T[], val: T): T[] {
  return arr.includes(val) ? arr.filter((v) => v !== val) : [...arr, val];
}

/** Type a ward name/id, pick one, the map flies there AND the queue narrows
 *  to it. Mirrors citizen/page.tsx's search-by-name pattern for consistency. */
function WardSearch({
  wardCells, activeWardId, onPick, onClear,
}: {
  wardCells: Ward[];
  activeWardId: string | null;
  onPick: (wardId: string) => void;
  onClear: () => void;
}) {
  const [q, setQ] = useState("");

  const wards = useMemo(() => {
    const seen = new Set<string>();
    const out: Ward[] = [];
    for (const w of wardCells) {
      if (!w.ward_id || w.ward_id === "unassigned" || seen.has(w.ward_id)) continue;
      seen.add(w.ward_id);
      out.push(w);
    }
    return out.sort((a, b) => a.ward_name.localeCompare(b.ward_name));
  }, [wardCells]);

  const hits = useMemo(() => {
    const query = q.trim().toLowerCase();
    if (!query) return [];
    return wards
      .filter((w) => w.ward_name.toLowerCase().includes(query) || w.ward_id.toLowerCase().includes(query))
      .slice(0, 8);
  }, [wards, q]);

  const activeWard = activeWardId ? wards.find((w) => w.ward_id === activeWardId) : null;

  return (
    <div>
      <div className="section-label" style={{ marginBottom: 6 }}>Ward</div>
      {activeWard ? (
        <div
          style={{
            display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6,
            padding: "5px 8px", borderRadius: "var(--radius-sm)",
            background: "var(--accent-soft)", border: "1px solid var(--accent-line)",
          }}
        >
          <span className="truncate" style={{ fontSize: "0.78rem", fontWeight: 550, color: "var(--text-primary)" }}>
            {activeWard.ward_name}
          </span>
          <button
            onClick={onClear}
            aria-label={`Stop filtering to ${activeWard.ward_name}`}
            style={{ display: "flex", background: "none", border: "none", cursor: "pointer", padding: 2, color: "var(--text-tertiary)" }}
          >
            <X {...icon.sm} aria-hidden />
          </button>
        </div>
      ) : (
        <div style={{ position: "relative" }}>
          <Search
            {...icon.sm}
            aria-hidden
            style={{ position: "absolute", left: 8, top: "50%", transform: "translateY(-50%)", color: "var(--text-tertiary)", pointerEvents: "none" }}
          />
          <input
            type="search"
            aria-label="Search for a ward"
            placeholder="Search wards…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            style={{ width: "100%", padding: "5px 8px 5px 26px", fontSize: "0.78rem" }}
          />
          {hits.length > 0 && (
            <div
              role="listbox"
              aria-label="Matching wards"
              className="menu"
              style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 1 }}
            >
              {hits.map((w) => (
                <button
                  key={w.ward_id}
                  role="option"
                  aria-selected={false}
                  className="menu-item"
                  onClick={() => { onPick(w.ward_id); setQ(""); }}
                >
                  <span className="truncate">{w.ward_name}</span>
                  <span className="menu-meta mono">{w.ward_id}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function FilterBar({ filters, wardCells = [], onWards, onSources, onPersistence, onFocusWard, onReset }: Props) {
  const activeCount = filters.ward_ids.length + filters.source_types.length + filters.persistence_types.length;

  return (
    <div
      className="glass"
      style={{
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
        minWidth: 176,
        boxShadow: "var(--shadow-md)",
      }}
    >
      <div
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "8px 12px", borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <span className="section-label">
          Filters{activeCount > 0 && ` · ${activeCount}`}
        </span>
        {activeCount > 0 && (
          <button
            onClick={onReset}
            style={{
              background: "none", border: "none", cursor: "pointer", padding: 0,
              fontSize: "0.7rem", color: "var(--accent)", fontFamily: "inherit",
              fontWeight: 520,
            }}
          >
            Clear
          </button>
        )}
      </div>

      <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 12 }}>
        <WardSearch
          wardCells={wardCells}
          activeWardId={filters.ward_ids[0] ?? null}
          onPick={(id) => { onWards([id]); onFocusWard?.(id); }}
          onClear={() => onWards([])}
        />
        <ChipGroup
          label="Persistence"
          options={PERSISTENCE_OPTIONS}
          selected={filters.persistence_types}
          onToggle={(v) => onPersistence(toggle(filters.persistence_types, v))}
          labels={PERSISTENCE_LABELS}
          tints={PERSIST_TINT}
        />
        <ChipGroup
          label="Source type"
          options={SOURCE_OPTIONS}
          selected={filters.source_types}
          onToggle={(v) => onSources(toggle(filters.source_types, v))}
          labels={SOURCE_LABELS}
          tints={SOURCE_TINT}
        />
      </div>
    </div>
  );
}
