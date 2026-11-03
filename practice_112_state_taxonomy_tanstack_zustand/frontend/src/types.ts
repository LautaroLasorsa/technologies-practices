export type TaskStatus = 'todo' | 'in_progress' | 'review' | 'done';
export type TaskPriority = 'low' | 'medium' | 'high' | 'urgent';

export interface Task {
  id: string;
  title: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  version: number;
  assignee: string;
  created_at: string;
  updated_at: string;
}

export interface TaskFilters {
  status?: string;
  priority?: string;
  search?: string;
  cursor?: string | null;
}

export interface TaskListResponse {
  items: Task[];
  next_cursor: string | null;
  total: number;
}

export interface TaskPatchInput {
  taskId: string;
  status?: TaskStatus;
  priority?: TaskPriority;
  title?: string;
  version: number;
  simulateConflict?: boolean;
}

export interface ConflictErrorDetails {
  message: string;
  task_id: string;
  client_expected_version: number;
  server_current_version: number;
  server_task: Task;
}

export interface ConflictEvent {
  id: string;
  taskId: string;
  message: string;
  clientExpectedVersion: number;
  serverCurrentVersion: number;
  timestamp: string;
}
