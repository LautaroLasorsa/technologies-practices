"use client";

import { useEffect, useState } from "react";

interface LocalTimestampBadgeProps {
  /**
   * UTC ISO 8601 timestamp string from the internal telemetry service (e.g. "2026-11-03T12:00:00Z")
   */
  utcIsoString: string;
}

/**
 * EXERCISE 4: SSR Hydration Mismatch Elimination
 *
 * THE PROBLEM:
 * When this component is pre-rendered on the Node.js server during SSR, Node executes in UTC (or the cloud server's local time).
 * When the browser receives the HTML and hydrates it, `new Date().toLocaleString()` evaluates to the client user's local timezone
 * (e.g., "America/New_York", "Europe/London", or "Asia/Tokyo").
 *
 * Because server-rendered HTML text ("12:00:00 PM UTC") does NOT match client virtual DOM text ("8:00:00 AM EDT"),
 * React throws a Hydration Mismatch Warning (Error #418 / #423 in React 19) and throws away the server DOM!
 *
 * TODO(human) [Exercise 4]:
 * 1. Implement progressive enhancement using a `mounted` state flag (`useState(false)` + `useEffect(() => setMounted(true), [])`).
 * 2. On initial render (and SSR), render a stable UTC representation or fallback (e.g., `new Date(utcIsoString).toUTCString()`
 *    or an ISO string).
 * 3. Once mounted on the client (`mounted === true`), render the localized format (`new Date(utcIsoString).toLocaleString()`).
 * 4. Alternatively, apply `suppressHydrationWarning` on the specific text container if minor divergence is acceptable.
 */
export default function LocalTimestampBadge({ utcIsoString }: LocalTimestampBadgeProps) {
  // TODO(human) [Exercise 4]: Replace this naive implementation that causes hydration mismatch
  // with a hydration-safe mounted state check or progressive enhancement pattern.

  /*
   * NAIVE FLAWED IMPLEMENTATION (Causes React Hydration Warning):
   * During SSR on Node: renders e.g. "11/3/2026, 12:00:00 PM" (UTC)
   * During Browser hydration: renders e.g. "11/3/2026, 9:00:00 AM" (Local)
   */
  const date = new Date(utcIsoString);
  const formattedString = date.toLocaleString(); // <-- Hydration mismatch!

  return (
    <span
      className="badge badge-cyan"
      title={`Server UTC: ${utcIsoString}`}
      style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}
    >
      ⏱ Telemetry Synced: {formattedString}
    </span>
  );
}
