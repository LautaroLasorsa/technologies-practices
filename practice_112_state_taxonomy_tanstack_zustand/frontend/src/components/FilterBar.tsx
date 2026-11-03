import React from 'react';
import { useUrlFilters } from '../hooks/useUrlFilters';

export const FilterBar: React.FC = () => {
  const { filters, setFilters } = useUrlFilters();

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 12,
        alignItems: 'center',
        padding: '12px 16px',
        backgroundColor: '#1e293b',
        borderRadius: 8,
        border: '1px solid #334155',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <label style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>
          Search
        </label>
        <input
          type="text"
          placeholder="Filter by title / text..."
          value={filters.search || ''}
          onChange={(e) => setFilters({ search: e.target.value })}
          style={{ width: 220 }}
        />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <label style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>
          Status (URL)
        </label>
        <select
          value={filters.status || 'all'}
          onChange={(e) => setFilters({ status: e.target.value })}
        >
          <option value="all">All Statuses</option>
          <option value="todo">To Do</option>
          <option value="in_progress">In Progress</option>
          <option value="review">Review</option>
          <option value="done">Done</option>
        </select>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <label style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>
          Priority (URL)
        </label>
        <select
          value={filters.priority || 'all'}
          onChange={(e) => setFilters({ priority: e.target.value })}
        >
          <option value="all">All Priorities</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="urgent">Urgent</option>
        </select>
      </div>

      <div style={{ marginLeft: 'auto', alignSelf: 'flex-end' }}>
        <button
          onClick={() => setFilters({ status: 'all', priority: 'all', search: '', cursor: null })}
          style={{ fontSize: 12, color: '#94a3b8' }}
        >
          Clear Filters
        </button>
      </div>
    </div>
  );
};
