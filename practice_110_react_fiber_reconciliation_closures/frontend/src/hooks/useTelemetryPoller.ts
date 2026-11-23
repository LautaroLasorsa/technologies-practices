import { useState, useEffect, useRef} from "react";
import { TelemetryMetric } from "../types";

interface UseTelemetryPollerOptions {
  pollingIntervalMs?: number;
  alertThreshold: number; // Percentage (e.g. 70 for 70% CPU)
}

interface UseTelemetryPollerResult {
  currentMetric: TelemetryMetric | null;
  history: TelemetryMetric[];
  alertTriggered: boolean;
  isPolling: boolean;
  tickCount: number;
  clearAlert: () => void;
}

export function useTelemetryPoller({
  pollingIntervalMs = 1000,
  alertThreshold,
}: UseTelemetryPollerOptions): UseTelemetryPollerResult {
  const [currentMetric, setCurrentMetric] = useState<TelemetryMetric | null>(null);
  const [history, setHistory] = useState<TelemetryMetric[]>([]);
  const [alertTriggered, setAlertTriggered] = useState(false);
  const [isPolling, _] = useState(true);
  const [tickCount, setTickCount] = useState(0);
  const thresholdRef = useRef(alertThreshold);
  useEffect(()=>{
    thresholdRef.current = alertThreshold;
  }, [alertThreshold]);
  
  /**
   * FLAGGED BEHAVIOR (Exercise 2 - Stale Closure Trap):
   *
   * The effect below registers an interval on mount with an empty dependency array `[]`.
   * Because of lexical scoping, the function passed to setInterval captures the initial
   * render's scope snapshot:
   * 1. `history` is captured as an empty array `[]`. Every tick executes `[...[], tick]`,
   *    so `history` never accumulates more than 1 item!
   * 2. `alertThreshold` is captured as its initial value (e.g. 70). If the operator moves
   *    the slider to 40, the interval still evaluates against the stale snapshot of 70!
   * 3. `tickCount` is captured as 0, so `setTickCount(tickCount + 1)` always sets 1!
   */
  useEffect(() => {
    if (!isPolling) return;
    
    const timer = setInterval(async () => {
      try {
        const response = await fetch("http://localhost:8110/api/telemetry/tick");
        if (!response.ok) return;

        const tick: TelemetryMetric = await response.json();
        setCurrentMetric(tick);

        // TODO(human): Exercise 2
        //
        // Problem 1: `history` is frozen at the initial render's snapshot.
        // Fix: Replace direct variable access with a functional state update:
        //      setHistory(prev => [...prev, tick].slice(-10));
        //
        // Problem 2: `tickCount` is frozen at 0.
        // Fix: setTickCount(prev => prev + 1);
        //
        // Problem 3: `alertThreshold` is captured from the initial render.
        // Fix: Use a ref (`thresholdRef = useRef(alertThreshold)`) updated on every render:
        //      useEffect(() => { thresholdRef.current = alertThreshold; });
        //      and read `thresholdRef.current` inside the interval callback without
        //      needing to reset the interval timer!

        // --- BUGGY IMPLEMENTATION FOR DEMONSTRATION ---
        setTickCount(prev => prev+1); // Stale closure: always sets 0 + 1 = 1!
        setHistory(prev => [...prev,tick].slice(-10));
        if (tick.cpu_usage > thresholdRef.current ) {
          // Stale closure: evaluates against initial alertThreshold
          setAlertTriggered(true);
        }
        // ----------------------------------------------
      } catch (err) {
        console.error("Telemetry tick failed:", err);
      }
    }, pollingIntervalMs);

    return () => clearInterval(timer);
    // Notice empty dependency array: intentional to demonstrate stale closure
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPolling, pollingIntervalMs]);

  return {
    currentMetric,
    history,
    alertTriggered,
    isPolling,
    tickCount,
    clearAlert: () => setAlertTriggered(false),
  };
}
