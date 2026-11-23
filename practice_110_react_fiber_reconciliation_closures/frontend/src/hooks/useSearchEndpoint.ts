import { useState, useEffect } from "react";
import { SearchResponse, TaskItem } from "../types";

interface UseSearchEndpointResult {
  query: string;
  setQuery: (q: string) => void;
  results: TaskItem[];
  isSearching: boolean;
  lastResolvedQuery: string;
  resolvedLatencyMs: number | null;
  hasRaceConditionOccurred: boolean;
  error: string | null;
}

export function useSearchEndpoint(initialQuery = ""): UseSearchEndpointResult {
  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState<TaskItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [lastResolvedQuery, setLastResolvedQuery] = useState("");
  const [resolvedLatencyMs, setResolvedLatencyMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  /**
   * FLAGGED BEHAVIOR (Exercise 3 - Async Race Conditions):
   *
   * The backend `/api/tasks/search` intentionally injects inverted latency:
   * - Queries with <= 2 chars take 1200ms
   * - Queries with > 2 chars take 80ms
   *
   * If you type "drone" quickly:
   * 1. Query "d" starts -> scheduled to finish in 1200ms
   * 2. Query "drone" starts -> finishes in 80ms
   * 3. UI renders results for "drone"
   * 4. At 1200ms, query "d" resolves and overwrites the state with stale results!
   * Notice that `lastResolvedQuery` will show "d" while the input contains "drone"!
   */
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    setError(null);

    // TODO(human): Exercise 3
    //
    // 1. Instantiate an AbortController:
    //    const controller = new AbortController();
    //
    // 2. Pass its signal to fetch:
    //    fetch(..., { signal: controller.signal })
    //
    // 3. In the catch block, ignore AbortError so cancelled requests do not set error state:
    //    if (err instanceof DOMException && err.name === "AbortError") {
    //      return; // Request was aborted cleanly
    //    }
    //
    // 4. Return an effect cleanup function that aborts in-flight requests:
    //    return () => controller.abort();
    //
    // Extra guard: You can also use an `active = true` flag inside the effect scope
    // and set `active = false` in the cleanup function.

    const controller = new AbortController();


    // --- BUGGY IMPLEMENTATION FOR EXPERIMENTATION ---
    fetch(
      `http://localhost:8110/api/tasks/search?q=${encodeURIComponent(query)}&delay_mode=jitter`,
      {signal: controller.signal}
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        return res.json() as Promise<SearchResponse>;
      })
      .then((data) => {
        // Stale response will overwrite newer state!
        setResults(data.results);
        setLastResolvedQuery(data.query);
        setResolvedLatencyMs(data.latency_ms);
        setIsSearching(false);
      })
      .catch((err) => {
        if(err instanceof DOMException && err.name === "AbortError"){
          return;
        }
        setError(err.message);
        setIsSearching(false);
      });

    // BUG: No cleanup function returning controller.abort()!
    // -----------------------------------------------
    return () => controller.abort();
  }, [query]);

  // Detected race condition: user typed something longer, but current results belong to an earlier query
  const hasRaceConditionOccurred =
    Boolean(query) &&
    Boolean(lastResolvedQuery) &&
    query !== lastResolvedQuery &&
    !isSearching;

  return {
    query,
    setQuery,
    results,
    isSearching,
    lastResolvedQuery,
    resolvedLatencyMs,
    hasRaceConditionOccurred,
    error,
  };
}
