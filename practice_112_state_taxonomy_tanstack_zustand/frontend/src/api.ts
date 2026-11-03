import { Task, TaskFilters, TaskListResponse, TaskPatchInput, TaskStatus } from './types';

const BASE_URL = '/api';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export async function fetchTasks(filters: TaskFilters = {}): Promise<TaskListResponse> {
  const params = new URLSearchParams();
  if (filters.status && filters.status !== 'all') {
    params.set('status', filters.status);
  }
  if (filters.priority && filters.priority !== 'all') {
    params.set('priority', filters.priority);
  }
  if (filters.search && filters.search.trim()) {
    params.set('search', filters.search.trim());
  }
  if (filters.cursor) {
    params.set('cursor', filters.cursor);
  }

  const res = await fetch(`${BASE_URL}/tasks?${params.toString()}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new ApiError(res.status, `Failed to fetch tasks: ${res.statusText}`, errorData);
  }
  return res.json();
}

export async function fetchTask(taskId: string): Promise<Task> {
  const res = await fetch(`${BASE_URL}/tasks/${taskId}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new ApiError(res.status, `Failed to fetch task ${taskId}`, errorData);
  }
  return res.json();
}

export async function patchTask(input: TaskPatchInput): Promise<Task> {
  const params = new URLSearchParams();
  if (input.simulateConflict) {
    params.set('simulate_conflict', 'true');
  }

  const url = `${BASE_URL}/tasks/${input.taskId}?${params.toString()}`;
  const res = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      status: input.status,
      priority: input.priority,
      title: input.title,
      version: input.version,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new ApiError(
      res.status,
      errorData.detail?.message || `Failed to patch task ${input.taskId}`,
      errorData.detail
    );
  }
  return res.json();
}

export async function batchUpdateTasks(
  taskIds: string[],
  status: TaskStatus
): Promise<{ updated_count: number; items: Task[] }> {
  const res = await fetch(`${BASE_URL}/tasks/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds, status }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new ApiError(res.status, `Failed batch update`, errorData);
  }
  return res.json();
}

export async function resetDatabase(): Promise<void> {
  const res = await fetch(`${BASE_URL}/reset`, { method: 'POST' });
  if (!res.ok) {
    throw new ApiError(res.status, 'Failed to reset database');
  }
}
