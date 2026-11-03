import { useCallback, useEffect, useState } from 'react';
import { TaskFilters } from '../types';

/**
 * Hook to synchronize task board filters with the browser URLSearchParams.
 *
 * THEORETICAL CONTEXT:
 * URL State is bookmarkable, shareable, and survives page reloads.
 * Instead of storing search filters or active tabs in React component useState
 * or Zustand, storing them in URLSearchParams makes the browser URL the single
 * source of truth for location/view parameters.
 */
export function useUrlFilters() {
  const [filters, setFiltersState] = useState<TaskFilters>(() => {
    const params = new URLSearchParams(window.location.search);
    return {
      status: params.get('status') || 'all',
      priority: params.get('priority') || 'all',
      search: params.get('search') || '',
      cursor: params.get('cursor') || null,
    };
  });

  // Listen for browser back/forward popstate events
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      setFiltersState({
        status: params.get('status') || 'all',
        priority: params.get('priority') || 'all',
        search: params.get('search') || '',
        cursor: params.get('cursor') || null,
      });
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  /**
   * TODO(human): Exercise 1 - Part A (URL State Synchronization)
   *
   * CONTEXT & OBJECTIVE:
   * Implement setFilters to update both React state and the browser's URLSearchParams.
   * When a filter changes (e.g. status, priority, or search term):
   * 1. Create a new URLSearchParams instance from window.location.search.
   * 2. For each key in updates:
   *    - If the value is empty, 'all', or null, delete the param from URLSearchParams to keep URLs clean.
   *    - Otherwise, set the param in URLSearchParams.
   * 3. If filter values other than 'cursor' change, reset 'cursor' to null (reset pagination on filter change).
   * 4. Call `window.history.pushState({}, '', newRelativeUrl)` so users can use back/forward navigation.
   * 5. Update local state `setFiltersState(...)`.
   *
   * HINT:
   * Construct `const newParams = new URLSearchParams(window.location.search)`
   * Check if 'cursor' in updates, reset if not.
   * Serialize cleanly: `window.history.pushState({}, '', newUrl)`
   */
  const setFilters = useCallback((updates: Partial<TaskFilters>) => {
    // === EXERCISE 1 PART A - TODO(human) ===
    throw new Error('TODO(human): Exercise 1 - Part A: Implement URL state synchronization in useUrlFilters');
  }, [filters]);

  return { filters, setFilters };
}
