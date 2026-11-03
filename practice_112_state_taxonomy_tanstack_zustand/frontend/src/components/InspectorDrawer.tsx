import React, { useState } from 'react';
import { useBoardUiStore, useInspectorTaskId } from '../store/useBoardUiStore';
import { useTasksQuery } from '../hooks/useTasksQuery';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { useOptimisticTaskMutation } from '../hooks/useOptimisticTaskMutation';
import { TaskPriority, TaskStatus } from '../types';

interface InspectorDrawerProps {
  simulateConflictDefault: boolean;
}

export const InspectorDrawer: React.FC<InspectorDrawerProps> = ({ simulateConflictDefault }) => {
  const inspectorTaskId = useInspectorTaskId();
  const setInspectorTaskId = useBoardUiStore((s) => s.setInspectorTaskId);
  const { filters } = useUrlFilters();
  const { data } = useTasksQuery(filters);
  const mutation = useOptimisticTaskMutation();

  const [simulateConflict, setSimulateConflict] = useState(simulateConflictDefault);

  const task = data?.items.find((t) => t.id === inspectorTaskId);

  if (!inspectorTaskId || !task) {
    return null;
  }

  const handleUpdate = (status: TaskStatus, priority: TaskPriority) => {
    mutation.mutate({
      taskId: task.id,
      status,
      priority,
      version: task.version,
      simulateConflict,
    });
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: 380,
        height: '100vh',
        backgroundColor: '#1e293b',
        borderLeft: '1px solid #334155',
        padding: 24,
        zIndex: 100,
        boxShadow: '-10px 0 25px -5px rgba(0, 0, 0, 0.5)',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: 11, color: '#38bdf8', fontFamily: 'monospace' }}>
            {task.id} (Version {task.version})
          </span>
          <h3 style={{ fontSize: 16, fontWeight: 600, color: '#f8fafc', marginTop: 4 }}>
            Inspector Drawer
          </h3>
        </div>
        <button onClick={() => setInspectorTaskId(null)} style={{ padding: '4px 8px' }}>
          ✕
        </button>
      </div>

      <div style={{ padding: 12, backgroundColor: '#0f172a', borderRadius: 6, fontSize: 13 }}>
        <p style={{ fontWeight: 600, color: '#f8fafc', marginBottom: 6 }}>{task.title}</p>
        <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.4 }}>{task.description}</p>
        <div style={{ marginTop: 10, fontSize: 11, color: '#64748b' }}>
          Assignee: <strong style={{ color: '#cbd5e1' }}>{task.assignee}</strong>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 12, color: '#94a3b8' }}>Status Transition</label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          {(['todo', 'in_progress', 'review', 'done'] as TaskStatus[]).map((st) => (
            <button
              key={st}
              onClick={() => handleUpdate(st, task.priority)}
              disabled={task.status === st || mutation.isPending}
              style={{
                backgroundColor: task.status === st ? '#334155' : '#1e293b',
                borderColor: task.status === st ? '#3b82f6' : '#334155',
              }}
            >
              {st.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 12, color: '#94a3b8' }}>Priority Transition</label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          {(['low', 'medium', 'high', 'urgent'] as TaskPriority[]).map((pr) => (
            <button
              key={pr}
              onClick={() => handleUpdate(task.status, pr)}
              disabled={task.priority === pr || mutation.isPending}
              style={{
                backgroundColor: task.priority === pr ? '#334155' : '#1e293b',
                borderColor: task.priority === pr ? '#3b82f6' : '#334155',
              }}
            >
              {pr}
            </button>
          ))}
        </div>
      </div>

      <div
        style={{
          marginTop: 'auto',
          padding: 12,
          backgroundColor: '#451a1a',
          borderRadius: 8,
          border: '1px solid #7f1d1d',
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={simulateConflict}
            onChange={(e) => setSimulateConflict(e.target.checked)}
          />
          <span style={{ fontSize: 12, color: '#fca5a5', fontWeight: 600 }}>
            Simulate 409 Conflict on Next Mutation
          </span>
        </label>
        <p style={{ fontSize: 11, color: '#f87171', marginTop: 4 }}>
          Forces the FastAPI backend to reject the patch with HTTP 409 Conflict. Observe the
          TanStack Query optimistic snapshot rollback!
        </p>
      </div>
    </div>
  );
};
