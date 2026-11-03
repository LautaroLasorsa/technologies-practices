import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { fetchTasks } from '../api';
import { TaskFilters, TaskListResponse } from '../types';

/**
 * Hierarchical Query Key Factory for Tasks.
 *
 * THEORETICAL CONTEXT:
 * Centralizing query keys in a structured factory prevents typo-induced cache misses
 * and enables precise invalidation scopes:
 * - Invalidate everything: taskKeys.all
 * - Invalidate all task lists: taskKeys.lists()
 * - Invalidate specific filtered query: taskKeys.list(filters)
 */
export const taskKeys = {
  all: ['tasks'] as const,
  lists: () => [...taskKeys.all, 'list'] as const,
  list: (filters: TaskFilters) => [...taskKeys.lists(), filters] as const,
  details: () => [...taskKeys.all, 'detail'] as const,
  detail: (id: string) => [...taskKeys.details(), id] as const,
};

/**
 * Hook to fetch tasks driven by URL filters and cached by TanStack Query.
 */
export function useTasksQuery(filters: TaskFilters) {
  /**
   * TODO(human): Exercise 1 - Part B (Hierarchical Query Key & Cache Configuration)
   *
   * CONTEXT & OBJECTIVE:
   * Connect TanStack Query's `useQuery` to fetchTasks using the hierarchical key factory.
   * Configure cache semantics:
   * 1. queryKey: derive using `taskKeys.list(filters)`. Notice how filter changes automatically
   *    trigger query evaluation because the key tuple changes.
   * 2. queryFn: call `() => fetchTasks(filters)`.
   * 3. staleTime: set to 10,000 ms (10s). During this window, component remounts or window focuses
   *    serve immediately from cache without background re-fetching.
   * 4. gcTime: set to 5 * 60 * 1000 ms (5 mins). Inactive query cache entries remain in memory
   *    for 5 minutes before being garbage collected.
   * 5. placeholderData: set to `keepPreviousData` from `@tanstack/react-query` to eliminate UI
   *    flicker when changing filter or cursor pages.
   *
   * HINT:
   * return useQuery({
   *   queryKey: ...,
   *   queryFn: ...,
   *   staleTime: 10_000,
   *   gcTime: 5 * 60_000,
   *   placeholderData: keepPreviousData,
   * });
   */

  // === EXERCISE 1 PART B - TODO(human) ===
  // Replace this placeholder with useQuery(...)
  return useQuery<TaskListResponse>({
    queryKey: ['TODO_EXERCISE_1_PART_B', filters],
    queryFn: () => {
      throw new Error('TODO(human): Exercise 1 - Part B: Configure useTasksQuery with taskKeys.list and cache options');
    },
  });
}
