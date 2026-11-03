import { create } from 'zustand';
import { ConflictEvent } from '../types';

export interface BoardUiState {
  selectedTaskIds: string[];
  inspectorTaskId: string | null;
  conflictEvents: ConflictEvent[];

  // Actions
  toggleTaskSelection: (taskId: string) => void;
  selectAllTasks: (taskIds: string[]) => void;
  clearSelection: () => void;
  setInspectorTaskId: (taskId: string | null) => void;
  addConflictEvent: (event: ConflictEvent) => void;
  dismissConflictEvent: (id: string) => void;
}

/**
 * Zustand Store for Client UI State.
 *
 * THEORETICAL CONTEXT:
 * Global UI state (selected cards, open modals/drawers, active toasts) is:
 * - Ephemeral to the user's browser session.
 * - Synchronously managed (no asynchronous fetch or cache invalidation needed).
 * - High-frequency updates (selection toggles, hover, drag).
 *
 * In classic React Context, any state change triggers re-renders on ALL consuming
 * components. Zustand uses `useSyncExternalStore` with selector subscriptions:
 * `useBoardUiStore(state => state.selectedTaskIds.includes(taskId))`
 * Components only re-render if the return value of their specific selector changes.
 */

/**
 * TODO(human): Exercise 3 (Zustand Ephemeral UI State & Atomic Selectors)
 *
 * CONTEXT & OBJECTIVE:
 * Implement the Zustand store using `create<BoardUiState>()((set) => ({ ... }))`.
 * 1. Initial state:
 *    selectedTaskIds: [],
 *    inspectorTaskId: null,
 *    conflictEvents: [],
 * 2. Actions:
 *    - toggleTaskSelection: if taskId is in selectedTaskIds, remove it; otherwise, append it.
 *    - selectAllTasks: set selectedTaskIds to unique array of taskIds.
 *    - clearSelection: set selectedTaskIds to empty array [].
 *    - setInspectorTaskId: set inspectorTaskId to given id or null.
 *    - addConflictEvent: prepend the event to conflictEvents list.
 *    - dismissConflictEvent: filter out event with matching id.
 */

// === EXERCISE 3 - TODO(human) ===
export const useBoardUiStore = create<BoardUiState>()((set) => ({
  selectedTaskIds: [],
  inspectorTaskId: null,
  conflictEvents: [],

  toggleTaskSelection: (taskId: string) => {
    throw new Error('TODO(human): Exercise 3: Implement toggleTaskSelection in useBoardUiStore');
  },
  selectAllTasks: (taskIds: string[]) => {
    throw new Error('TODO(human): Exercise 3: Implement selectAllTasks in useBoardUiStore');
  },
  clearSelection: () => {
    throw new Error('TODO(human): Exercise 3: Implement clearSelection in useBoardUiStore');
  },
  setInspectorTaskId: (taskId: string | null) => {
    throw new Error('TODO(human): Exercise 3: Implement setInspectorTaskId in useBoardUiStore');
  },
  addConflictEvent: (event: ConflictEvent) => {
    throw new Error('TODO(human): Exercise 3: Implement addConflictEvent in useBoardUiStore');
  },
  dismissConflictEvent: (id: string) => {
    throw new Error('TODO(human): Exercise 3: Implement dismissConflictEvent in useBoardUiStore');
  },
}));

// === ATOMIC SELECTORS FOR HIGH PERFORMANCE RENDERING ===

/** Subscribes ONLY to whether a specific task is selected. */
export function useIsTaskSelected(taskId: string): boolean {
  return useBoardUiStore((s) => s.selectedTaskIds.includes(taskId));
}

/** Subscribes ONLY to the count of selected tasks. */
export function useSelectedTaskCount(): number {
  return useBoardUiStore((s) => s.selectedTaskIds.length);
}

/** Subscribes ONLY to inspector task id. */
export function useInspectorTaskId(): string | null {
  return useBoardUiStore((s) => s.inspectorTaskId);
}
