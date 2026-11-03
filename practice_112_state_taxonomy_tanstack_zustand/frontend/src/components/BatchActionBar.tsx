import React from 'react';
import { useBoardUiStore, useSelectedTaskCount } from '../store/useBoardUiStore';
import { TaskStatus } from '../types';

interface BatchActionBarProps {
  onExecuteBatch: (status: TaskStatus) => void;
  isPending: boolean;
}

export const BatchActionBar: React.FC<BatchActionBarProps> = ({ onExecuteBatch, isPending }) => {
  // Atomic selector: only re-renders when count changes, not on card clicks that keep count constant
  const selectedCount = useSelectedTaskCount();
  const clearSelection = useBoardUiStore((s) => s.clearSelection);

  if (selectedCount === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: '#1e293b',
        border: '1px solid #3b82f6',
        borderRadius: 12,
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5)',
        zIndex: 50,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span
          style={{
            backgroundColor: '#3b82f6',
            color: '#fff',
            borderRadius: 9999,
            padding: '2px 8px',
            fontSize: 12,
            fontWeight: 700,
          }}
        >
          {selectedCount}
        </span>
        <span style={{ fontSize: 13, fontWeight: 500, color: '#f8fafc' }}>
          selected (Zustand client state)
        </span>
      </div>

      <div style={{ height: 20, width: 1, backgroundColor: '#334155' }} />

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={() => onExecuteBatch('in_progress')}
          disabled={isPending}
          className="btn-primary"
        >
          Mark In Progress
        </button>
        <button
          onClick={() => onExecuteBatch('done')}
          disabled={isPending}
          style={{ backgroundColor: '#10b981', borderColor: '#10b981', color: '#fff' }}
        >
          Mark Done
        </button>
        <button onClick={clearSelection} disabled={isPending}>
          Clear
        </button>
      </div>
    </div>
  );
};
