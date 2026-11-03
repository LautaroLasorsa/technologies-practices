import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { taskKeys, useTasksQuery } from '../hooks/useTasksQuery';
import { useBoardUiStore } from '../store/useBoardUiStore';
import { batchUpdateTasks } from '../api';
import { TaskCard } from './TaskCard';
import { FilterBar } from './FilterBar';
import { BatchActionBar } from './BatchActionBar';
import { InspectorDrawer } from './InspectorDrawer';
import { ConflictSimulatorControl } from './ConflictSimulatorControl';
import { TaskStatus } from '../types';

export const BoardWorkspace: React.FC = () => {
  const queryClient = useQueryClient();
  const { filters, setFilters } = useUrlFilters();
  const { data, isLoading, isError, error, isPlaceholderData } = useTasksQuery(filters);

  const selectedTaskIds = useBoardUiStore((s) => s.selectedTaskIds);
  const selectAllTasks = useBoardUiStore((s) => s.selectAllTasks);
  const clearSelection = useBoardUiStore((s) => s.clearSelection);
  const setInspectorTaskId = useBoardUiStore((s) => s.setInspectorTaskId);

  const [simulateConflictGlobal, setSimulateConflictGlobal] = useState(false);

  /**
   * TODO(human): Exercise 4 (Combining Server State & Zustand Client State)
   *
   * CONTEXT & OBJECTIVE:
   * When performing batch actions across multiple selected items:
   * 1. The selection array (`selectedTaskIds`) lives in Zustand client state.
   * 2. The batch mutation must be dispatched to the backend via `batchUpdateTasks(selectedTaskIds, targetStatus)`.
   * 3. Once the server accepts the batch update:
   *    a. Invalidate the TanStack Query cache (`queryClient.invalidateQueries({ queryKey: taskKeys.all })`)
   *       so all task cards reflect their new server status and updated versions.
   *    b. Clear the Zustand selection (`clearSelection()`) so stale client IDs do not persist.
   *
   * Configure this `useMutation` hook to fulfill these steps.
   */
  const batchMutation = useMutation({
    mutationFn: async (targetStatus: TaskStatus) => {
      // === EXERCISE 4 - TODO(human) ===
      throw new Error('TODO(human): Exercise 4: Implement batch mutation combining Zustand and TanStack Query');
    },
  });

  const handleSelectAllOnPage = () => {
    if (data?.items) {
      selectAllTasks(data.items.map((t) => t.id));
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 20px 100px 20px', display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 12, padding: '2px 8px', borderRadius: 4, backgroundColor: '#3b82f6', color: '#fff', fontWeight: 600 }}>
              Practice 112
            </span>
            <span style={{ fontSize: 13, color: '#94a3b8' }}>
              Operations Task Board
            </span>
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: '#f8fafc' }}>
            State Taxonomy: TanStack Query vs Zustand vs URL
          </h1>
          <p style={{ fontSize: 14, color: '#94a3b8', marginTop: 4 }}>
            Server Cache (asynchronous, versioned) · Global UI State (synchronous, atomic) · URL State (bookmarkable)
          </p>
        </div>
      </header>

      {/* Simulator Control & Conflict Toasts */}
      <ConflictSimulatorControl
        simulateConflictGlobal={simulateConflictGlobal}
        onToggleGlobalConflict={setSimulateConflictGlobal}
      />

      {/* URL Filters */}
      <FilterBar />

      {/* Board Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={handleSelectAllOnPage} style={{ fontSize: 12 }}>
            Select All on Page
          </button>
          {selectedTaskIds.length > 0 && (
            <button onClick={clearSelection} style={{ fontSize: 12, color: '#94a3b8' }}>
              Deselect All ({selectedTaskIds.length})
            </button>
          )}
        </div>

        <div style={{ fontSize: 13, color: '#94a3b8' }}>
          Total Tasks in DB: <strong style={{ color: '#f8fafc' }}>{data?.total ?? '—'}</strong>
          {isPlaceholderData && <span style={{ marginLeft: 8, color: '#f59e0b' }}>(cached preview)</span>}
        </div>
      </div>

      {/* Loading & Error States */}
      {isLoading && (
        <div style={{ padding: 48, textAlign: 'center', color: '#94a3b8', backgroundColor: '#1e293b', borderRadius: 8 }}>
          Loading tasks from FastAPI server state...
        </div>
      )}

      {isError && (
        <div style={{ padding: 24, backgroundColor: '#451a1a', border: '1px solid #7f1d1d', borderRadius: 8, color: '#fca5a5' }}>
          <strong>Error loading tasks:</strong> {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      )}

      {/* Tasks Grid */}
      {data && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: 16,
          }}
        >
          {data.items.map((task) => (
            <TaskCard
              key={task.id}
              task={task}
              onOpenInspector={(id) => setInspectorTaskId(id)}
              simulateConflictNext={simulateConflictGlobal}
            />
          ))}
        </div>
      )}

      {/* Pagination Controls */}
      {data?.next_cursor && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 12 }}>
          <button
            onClick={() => setFilters({ cursor: data.next_cursor })}
            className="btn-primary"
            style={{ padding: '8px 20px', fontSize: 14 }}
          >
            Load Next Page (Cursor: {data.next_cursor})
          </button>
        </div>
      )}

      {/* Sticky Batch Action Bar (Zustand client state driver) */}
      <BatchActionBar
        onExecuteBatch={(status) => batchMutation.mutate(status)}
        isPending={batchMutation.isPending}
      />

      {/* Inspector Side Drawer */}
      <InspectorDrawer simulateConflictDefault={simulateConflictGlobal} />
    </div>
  );
};
