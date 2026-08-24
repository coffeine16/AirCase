"use client";
import { use, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useReports } from "@/hooks/useReports";
import { REPORT_CATEGORY_LABELS, REPORT_CATEGORY_ICONS } from "@/lib/constants";
import { icon, ArrowLeft, Camera, CircleCheck, LoaderCircle, TriangleAlert } from "@/components/Icon";
import type { ReportCategory, CreateReportPayload } from "@/lib/types";

interface Params { wardId: string }

const CATEGORIES: ReportCategory[] = [
  "waste_burning", "construction_dust", "industrial", "traffic", "other"
];

export default function ReportPage({ params }: { params: Promise<Params> }) {
  const { wardId } = use(params);
  const router = useRouter();
  const { submitReport, submitting, submitError } = useReports();

  const [category, setCategory] = useState<ReportCategory | null>(null);
  const [description, setDescription] = useState("");
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const handlePhoto = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPhoto(file);
    setPhotoPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!category) return;
    const payload: CreateReportPayload = {
      ward_id: wardId,
      category,
      description: description.trim() || undefined,
      photo: photo ?? undefined,
    };
    const result = await submitReport(payload);
    if (result) setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="page" style={{ maxWidth: 520, textAlign: "center", paddingTop: "var(--space-2xl)" }}>
        <CircleCheck
          {...icon.lg}
          aria-hidden
          style={{ color: "var(--positive)", marginBottom: "var(--space-md)" }}
        />
        <h2 style={{ marginBottom: "var(--space-sm)" }}>Report submitted</h2>
        <p style={{ marginBottom: "var(--space-xl)" }}>
          Report received. Track it in My Reports.
        </p>
        <div style={{ display: "flex", gap: "var(--space-md)", justifyContent: "center", flexWrap: "wrap" }}>
          <Link href="/citizen/reports" className="btn btn-primary" style={{ textDecoration: "none" }}>
            View my reports
          </Link>
          <Link href={`/citizen/${wardId}`} className="btn btn-ghost" style={{ textDecoration: "none" }}>
            <ArrowLeft {...icon.sm} aria-hidden />
            Back to ward
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="page" style={{ maxWidth: 560 }}>
      <Link href={`/citizen/${wardId}`} className="nav-link" style={{ marginBottom: "var(--space-md)", marginLeft: -10 }}>
        <ArrowLeft {...icon.sm} aria-hidden />
        Back to ward
      </Link>

      <h1 style={{ marginBottom: 6 }}>Report pollution</h1>
      <p style={{ marginBottom: "var(--space-xl)" }}>
        Help us identify pollution sources in {wardId}. Your report becomes evidence.
      </p>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "var(--space-lg)" }}>
        {/* Category */}
        <div>
          <label>What type of pollution did you see? *</label>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "var(--space-sm)", marginTop: 8 }}>
            {CATEGORIES.map((cat) => {
              const Glyph = REPORT_CATEGORY_ICONS[cat];
              const on = category === cat;
              return (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setCategory(cat)}
                  className="card card-hover"
                  aria-pressed={on}
                  style={{
                    cursor: "pointer",
                    textAlign: "left",
                    display: "flex",
                    alignItems: "center",
                    gap: 9,
                    padding: "11px 12px",
                    fontFamily: "inherit",
                    borderColor: on ? "var(--accent)" : undefined,
                    background: on ? "var(--accent-soft)" : undefined,
                  }}
                >
                  <Glyph
                    {...icon.md}
                    aria-hidden
                    style={{ color: on ? "var(--accent)" : "var(--text-tertiary)", flexShrink: 0 }}
                  />
                  <span style={{ fontSize: "0.8rem", fontWeight: 500, color: "var(--text-primary)" }}>
                    {REPORT_CATEGORY_LABELS[cat]}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Photo */}
        <div>
          <label>Photo (optional)</label>
          <div
            style={{
              marginTop: 8,
              border: "1px dashed var(--border-default)",
              borderRadius: "var(--radius-lg)",
              padding: "var(--space-lg)",
              textAlign: "center",
              cursor: "pointer",
              transition: "border-color var(--transition-fast)",
              position: "relative",
            }}
            onClick={() => document.getElementById("photo-input")?.click()}
          >
            {photoPreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={photoPreview}
                alt="Preview"
                style={{ maxHeight: 200, maxWidth: "100%", objectFit: "contain", borderRadius: 6 }}
              />
            ) : (
              <div style={{ color: "var(--text-tertiary)", display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
                <Camera {...icon.lg} aria-hidden />
                <div style={{ fontSize: "0.8125rem" }}>Tap to add a photo</div>
              </div>
            )}
            <input
              id="photo-input"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handlePhoto}
              style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer" }}
            />
          </div>
        </div>

        {/* Description */}
        <div>
          <label>Description (optional)</label>
          <textarea
            rows={3}
            placeholder="What did you see? Where exactly? Any other details..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            style={{ resize: "vertical" }}
          />
        </div>

        {/* Error */}
        {submitError && (
          <div role="alert" className="alert alert-critical">
            <TriangleAlert {...icon.md} aria-hidden />
            <div className="alert-body">
              <strong>Report not submitted</strong>
              {submitError}
            </div>
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!category || submitting}
          style={{ width: "100%", padding: "11px" }}
        >
          {submitting ? (
            <>
              <LoaderCircle {...icon.md} className="animate-spin" aria-hidden />
              Submitting…
            </>
          ) : "Submit report"}
        </button>
      </form>
    </div>
  );
}
