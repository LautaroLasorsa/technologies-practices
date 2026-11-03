import React from 'react';
import { useBoardUiStore } from '../store/useBoardUiStore';
import { resetDatabase } from '../api';
import { useQueryClient } from '@tanstack/react-query';
import { taskKeys } from '../hooks/useTasksQuery';

interface ConflictSimulatorControlProps {
  simulateConflictGlobal: boolean;
  onToggleGlobalConflict: (enabled: boolean) => void;
}

export const ConflictSimulatorControl: React.FC<ConflictSimulatorControlProps> = ({
  simulateConflictGlobal,
  onToggleGlobalConflict,
}) => {
  const queryClient = useQueryClient();
  const conflictEvents = useBoardUiStore((s) => s.conflictEvents);
  const dismissConflictEvent = useBoardUiStore((s) => s.dismissConflictEvent);

  const handleReset = async () => {
    try {
      await resetDatabase();
      await queryClient.invalidateQueries({ queryKey: taskKeys.all });
    } catch (err) {
      console.error('Reset failed', err);
    }
  };

  return (
    <>
      <div
        style={{
          backgroundColor: '#1e293b',
          border: '1px solid #334155',
          borderRadius: 8,
          padding: '12px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={simulateConflictGlobal}
              onChange={(e) => onToggleGlobalConflict(e.target.checked)}
            />
            <span style={{ fontSize: 13, color: '#fca5a5', fontWeight: 600 }}>
              Simulate 409 Conflict (All Card Mutations)
            </span>
          </label>
          <span style={{ fontSize: 12, color: '#64748b' }}>|</span>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            Tests optimistic update rollback and version mismatch handling.
          </span>
        </div>

        <button onClick={handleReset} style={{ fontSize: 12 }}>
          Reset Backend Seed Data
        </button>
      </div>

      {/* Floating Conflict Toasts */}
      {conflictEvents.length > 0 && (
        <div className="toast-container">
          {conflictEvents.map((evt) => (
            <div key={evt.id} className="toast">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ color: '#f87171' }}>
                  409 Conflict Rollback: {evt.taskId}
                </strong>
                <button
                  onClick={() => dismissConflictEvent(evt.id)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#94a3b8',
                    padding: 0,
                    fontSize: 14,
                  }}
                >
                  ✕
                </button>
              </div>
              <p style={{ marginTop: 4, color: '#cbd5e1' }}>{evt.message}</p>
              <div
                style={{
                  marginTop: 6,
                  fontSize: 11,
                  fontFamily: 'monospace',
                  color: '#94a3b8',
                }}
              >
                Expected: v{evt.clientExpectedVersion} | Server: v{evt.serverCurrentVersion} | {evt.timestamp}
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
};
