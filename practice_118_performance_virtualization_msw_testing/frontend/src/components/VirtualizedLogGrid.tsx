import React, { useRef, useDeferredValue } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { LogRecord, LogLevel } from '../types';

export interface VirtualizedLogGridProps {
  logs: LogRecord[];
  naiveMode?: boolean; // When true, renders unvirtualized for comparison benchmarking
}

const SEVERITY_BADGE_STYLES: Record<LogLevel, { bg: string; text: string }> = {
  INFO: { bg: '#1e293b', text: '#94a3b8' },
  WARN: { bg: '#854d0e', text: '#fef08a' },
  ERROR: { bg: '#991b1b', text: '#fecaca' },
  CRITICAL: { bg: '#701a75', text: '#f5d0fe' },
};

export const VirtualizedLogGrid: React.FC<VirtualizedLogGridProps> = ({
  logs,
  naiveMode = false,
}) => {
  const parentRef = useRef<HTMLDivElement>(null);

  // =========================================================================
  // TODO(human): Exercise 2 - Configure TanStack Virtualizer & useDeferredValue
  // =========================================================================
  // Instructions:
  // 1. Configure useVirtualizer with:
  //    - count: logs.length
  //    - getScrollElement: () => parentRef.current
  //    - estimateSize: () => 42 (pixel height of each row)
  //    - overscan: 10 (render 10 items above and below visible viewport)
  // 2. Wrap virtualizer.getVirtualItems() with React's useDeferredValue so that
  //    rapid scroll events do not block critical UI updates.
  // 3. Render a phantom spacer container sized to virtualizer.getTotalSize() px.
  // 4. Render virtualItems absolutely positioned using `transform: translateY(${item.start}px)`.
  const rowVirtualizer = useVirtualizer({
    count: logs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 42,
    overscan: 10,
  });

  const virtualItems = useDeferredValue(rowVirtualizer.getVirtualItems());
  const totalSize = rowVirtualizer.getTotalSize();

  if (naiveMode) {
    // Naive rendering mode: renders all DOM nodes directly (will freeze on 50,000 items)
    return (
      <div
        ref={parentRef}
        style={{
          height: '600px',
          overflowY: 'auto',
          border: '1px solid #334155',
          borderRadius: '8px',
          backgroundColor: '#090d16',
        }}
        data-testid="log-grid-container"
      >
        <div style={{ padding: '8px', color: '#f59e0b', fontSize: '12px' }}>
          ⚠️ Naive Mode: Rendering all {logs.length} DOM elements directly (Layout / Paint cost: High)
        </div>
        {logs.map((log) => (
          <LogRow key={log.id} log={log} />
        ))}
      </div>
    );
  }

  return (
    <div
      ref={parentRef}
      style={{
        height: '600px',
        overflowY: 'auto',
        border: '1px solid #334155',
        borderRadius: '8px',
        backgroundColor: '#090d16',
        position: 'relative',
      }}
      data-testid="log-grid-container"
    >
      {/* Phantom spacer maintaining full scrollable height */}
      <div
        style={{
          height: `${totalSize}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualItems.map((virtualRow) => {
          const log = logs[virtualRow.index];
          if (!log) return null;

          return (
            <div
              key={virtualRow.key}
              data-index={virtualRow.index}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <LogRow log={log} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

const LogRow: React.FC<{ log: LogRecord }> = React.memo(({ log }) => {
  const badge = SEVERITY_BADGE_STYLES[log.level];

  return (
    <div
      data-testid={`log-row-${log.id}`}
      style={{
        display: 'grid',
        gridTemplateColumns: '90px 140px 180px 1fr 80px 70px',
        alignItems: 'center',
        padding: '0 12px',
        height: '42px',
        borderBottom: '1px solid #1e293b',
        fontSize: '13px',
        fontFamily: 'monospace',
      }}
    >
      <span
        style={{
          display: 'inline-block',
          textAlign: 'center',
          padding: '2px 6px',
          borderRadius: '4px',
          backgroundColor: badge.bg,
          color: badge.text,
          fontWeight: 'bold',
          fontSize: '11px',
        }}
      >
        {log.level}
      </span>
      <span style={{ color: '#64748b' }}>
        {log.timestamp.slice(11, 23)}
      </span>
      <span style={{ color: '#38bdf8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {log.service}
      </span>
      <span style={{ color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {log.message}
      </span>
      <span style={{ color: log.latencyMs > 300 ? '#f87171' : '#94a3b8', textAlign: 'right' }}>
        {log.latencyMs}ms
      </span>
      <span
        style={{
          textAlign: 'right',
          color: log.anomalyScore > 0.7 ? '#ef4444' : '#10b981',
          fontWeight: log.anomalyScore > 0.7 ? 'bold' : 'normal',
        }}
      >
        {log.anomalyScore.toFixed(2)}
      </span>
    </div>
  );
});
