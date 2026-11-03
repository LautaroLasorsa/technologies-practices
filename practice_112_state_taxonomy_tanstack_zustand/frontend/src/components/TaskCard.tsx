import React, { useRef } from 'react';
import { Task, TaskPriority, TaskStatus } from '../types';
import { useBoardUiStore, useIsTaskSelected } from '../store/useBoardUiStore';
import { useOptimisticTaskMutation } from '../hooks/useOptimisticTaskMutation';

interface TaskCardProps {
  task: Task;
  onOpenInspector: (taskId: string) => void;
  simulateConflictNext: boolean;
}

export const TaskCard: React.FC<TaskCardProps> = React.memo(
  ({ task, onOpenInspector, simulateConflictNext }) => {
    // Atomic selector: only re-renders when THIS card's selection state toggles
    const isSelected = useIsTaskSelected(task.id);
    const toggleTaskSelection = useBoardUiStore((s) => s.toggleTaskSelection);
    const mutation = useOptimisticTaskMutation();

    // Render count indicator to visually prove re-render isolation
    const renderCount = useRef(0);
    renderCount.current += 1;

    const handleQuickStatusAdvance = (e: React.MouseEvent) => {
      e.stopPropagation();
      const statusCycle: Record<TaskStatus, TaskStatus> = {
        todo: 'in_progress',
        in_progress: 'review',
        review: 'done',
        done: 'todo',
      };
      const nextStatus = statusCycle[task.status];

      mutation.mutate({
        taskId: task.id,
        status: nextStatus,
        version: task.version,
        simulateConflict: simulateConflictNext,
      });
    };

    return (
      <div
        onClick={() => onOpenInspector(task.id)}
        style={{
          backgroundColor: isSelected ? '#1e293b' : '#182030',
          border: isSelected ? '1px solid #3b82f6' : '1px solid #334155',
          borderRadius: 8,
          padding: 14,
          cursor: 'pointer',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          boxShadow: isSelected ? '0 0 0 1px #3b82f6' : 'none',
          transition: 'all 0.15s ease',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input
              type="checkbox"
              checked={isSelected}
              onClick={(e) => e.stopPropagation()}
              onChange={() => toggleTaskSelection(task.id)}
              style={{ width: 16, height: 16, cursor: 'pointer' }}
              title="Select for batch operations (Zustand client state)"
            />
            <span style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace' }}>
              {task.id}
            </span>
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span
              style={{
                fontSize: 10,
                padding: '2px 6px',
                borderRadius: 4,
                backgroundColor: '#0f172a',
                color: '#38bdf8',
                border: '1px solid #0284c7',
                fontFamily: 'monospace',
              }}
              title="Optimistic Concurrency Version"
            >
              v{task.version}
            </span>
            <span
              style={{
                fontSize: 10,
                padding: '2px 5px',
                borderRadius: 4,
                backgroundColor: '#334155',
                color: '#94a3b8',
              }}
              title="Component render count (demonstrates selector isolation)"
            >
              R:{renderCount.current}
            </span>
          </div>
        </div>

        <div>
          <h4 style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginBottom: 4 }}>
            {task.title}
          </h4>
          <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.4 }}>
            {task.description}
          </p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
          <div style={{ display: 'flex', gap: 6 }}>
            <span className={`badge badge-${task.status}`}>{task.status.replace('_', ' ')}</span>
            <span className={`badge badge-${task.priority}`}>{task.priority}</span>
          </div>

          <button
            onClick={handleQuickStatusAdvance}
            disabled={mutation.isPending}
            style={{
              fontSize: 11,
              padding: '3px 8px',
              backgroundColor: '#334155',
              border: 'none',
              borderRadius: 4,
              color: '#f8fafc',
            }}
            title="Advance status optimistically (Exercise 2)"
          >
            {mutation.isPending ? 'Advancing...' : 'Advance >'}
          </button>
        </div>
      </div>
    );
  }
);
