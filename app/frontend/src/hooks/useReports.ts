"use client";
import useSWR from "swr";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import type { CitizenReport, CreateReportPayload } from "@/lib/types";

const WARD_KEY = "aq_citizen_ward";
const DEVICE_KEY = "aq_citizen_device_id";

export function useCitizenWard() {
  const [wardId, setWardIdState] = useState<string | null>(null);

  useEffect(() => {
    setWardIdState(localStorage.getItem(WARD_KEY));
  }, []);

  const setWardId = (id: string) => {
    localStorage.setItem(WARD_KEY, id);
    setWardIdState(id);
  };

  const clearWard = () => {
    localStorage.removeItem(WARD_KEY);
    setWardIdState(null);
  };

  return { wardId, setWardId, clearWard };
}

/**
 * Anonymous per-browser id — resolves the "Q2 open" identity question that
 * used to sit here. No account, no phone number, no name: a random UUID
 * minted once and kept in localStorage, sent with every report so `GET
 * /reports` can return one citizen's own reports instead of every report in
 * the city (Supabase RLS lets the anon key read all rows; this is what
 * narrows it down before anything reaches a browser). Losing localStorage
 * (a new device, a cleared browser) loses the report history along with
 * it — the same tradeoff `useCitizenWard` already makes for the saved ward,
 * and the only way to get history back without asking for a phone number.
 */
export function useDeviceId(): string | null {
  const [deviceId, setDeviceId] = useState<string | null>(null);

  useEffect(() => {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(DEVICE_KEY, id);
    }
    setDeviceId(id);
  }, []);

  return deviceId;
}

export function useReports() {
  const deviceId = useDeviceId();

  // Deferred (null key) until deviceId is ready, so the first request never
  // fires with an empty id and gets cached as "this device has no reports".
  const { data, error, isLoading, mutate } = useSWR<CitizenReport[]>(
    deviceId ? ["reports", deviceId] : null,
    () => api.getReports(deviceId!),
    { revalidateOnFocus: false }
  );

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const submitReport = async (payload: CreateReportPayload): Promise<CitizenReport | null> => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const report = await api.submitReport(payload, deviceId ?? "");
      // Optimistic update
      mutate((prev) => (prev ? [report, ...prev] : [report]), false);
      await mutate();
      return report;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Submission failed";
      setSubmitError(msg);
      return null;
    } finally {
      setSubmitting(false);
    }
  };

  return {
    reports: data ?? [],
    error,
    isLoading,
    submitting,
    submitError,
    submitReport,
    refresh: mutate,
  };
}
