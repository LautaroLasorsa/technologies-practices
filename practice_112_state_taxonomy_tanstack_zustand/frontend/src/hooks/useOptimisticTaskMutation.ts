import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ApiError, patchTask } from '../api';
import { TaskListResponse, TaskPatchInput } from '../types';
import { taskKeys } from './useTasksQuery';
import { useBoardUiStore } from '../store/useBoardUiStore';

interface MutationContext {
  previousQueries: [readonly unknown[], TaskListResponse | undefined][];
}

/**
 * Hook for mutating a task with optimistic cache updates and rollback on 409 Conflict.
 *
 * THEORETICAL CONTEXT:
 * Optimistic Updates provide zero-latency UI response. The browser applies the mutation
 * to its local memory cache before the network packet reaches the server.
 * However, in distributed systems with concurrent users, the server may reject the update
 * (e.g., HTTP 409 Conflict due to version mismatch).
 * A resilient optimistic update protocol requires:
 * 1. onMutate: Cancel inflight queries -> Take snapshot -> Patch cache -> Return snapshot context
 * 2. onError: Restore snapshot cache if mutation fails or conflicts
 * 3. onSettled: Trigger background revalidation to guarantee eventual consistency with server
 */
export function useOptimisticTaskMutation() {
  const queryClient = useQueryClient();
  const addConflictEvent = useBoardUiStore((s) => s.addConflictEvent);

  /**
   * TODO(human): Exercise 2 (Optimistic Update & 409 Conflict Rollback)
   *
   * CONTEXT & OBJECTIVE:
   * Implement `useMutation` with `onMutate`, `onError`, and `onSettled`:
   *
   * 1. mutationFn: (input: TaskPatchInput) => patchTask(input)
   *
   * 2. onMutate(input):
   *    a. Cancel outgoing refetches using `await queryClient.cancelQueries({ queryKey: taskKeys.all })`.
   *       This prevents inflight network responses from overwriting our optimistic state.
   *    b. Snapshot previous query data for all task lists using:
   *       `const previousQueries = queryClient.getQueriesData<TaskListResponse>({ queryKey: taskKeys.lists() })`
   *    c. Optimistically update all matching task list queries using `queryClient.setQueriesData`:
   *       Find the task by `input.taskId` and update its status, priority, title, and increment version by 1.
   *    d. Return `{ previousQueries }` as mutation context for rollback.
   *
   * 3. onError(error, input, context):
   *    a. If context?.previousQueries exists, restore every snapshot query via:
   *       `queryClient.setQueryData(queryKey, data)`
   *    b. If the error is an ApiError (HTTP 409 Conflict), record the conflict in Zustand:
   *       `addConflictEvent({
   *          id: crypto.randomUUID(),
   *          taskId: input.taskId,
   *          message: error.message,
   *          clientExpectedVersion: input.version,
   *          serverCurrentVersion: (error as ApiError).data?.server_current_version ?? input.version,
   *          timestamp: new Date().toLocaleTimeString(),
   *       })`
   *
   * 4. onSettled():
   *    a. Invalidate task queries so the client reconciles with canonical server state:
   *       `queryClient.invalidateQueries({ queryKey: taskKeys.all })`
   */

  return useMutation<unknown, Error, TaskPatchInput, MutationContext>({
    mutationFn: (input: TaskPatchInput) => {
      // === EXERCISE 2 - TODO(human) ===
      throw new Error('TODO(human): Exercise 2: Implement optimistic mutation in useOptimisticTaskMutation');
    },
  });
}
