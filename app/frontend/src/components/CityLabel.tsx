"use client";
import { useCity } from "@/lib/CityContext";

/** The current city's name — a client leaf so server layouts can show it live. */
export default function CityLabel({ style }: { style?: React.CSSProperties }) {
  const { cityLabel } = useCity();
  return <span style={style}>{cityLabel}</span>;
}
