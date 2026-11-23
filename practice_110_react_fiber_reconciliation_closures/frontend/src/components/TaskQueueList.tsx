import React, { useState } from "react";
import { TaskItem } from "../types";

interface TaskQueueListProps {
  tasks: TaskItem[];
  onTasksChange: (tasks: TaskItem[]) => void;
}

/**
 * TaskQueueItem renders individual task controls with an uncontrolled notes input.
 * When keys are unstable or index-based, Fiber reuses the DOM nodes and internal
 * component state based on index position, causing input contents to detach from
 * their parent task item during re-ordering.
 */
function TaskQueueItem({
  task,
  index,
  total,
  onMove,
}: {
  task: TaskItem;
  index: number;
  total: number;
  onMove: (index: number, direction: "up" | "down") => void;
}) {
  // Local state representing uncontrolled user note or selection
  const [operatorNote, setOperatorNote] = useState("");

  const priorityColors: Record<TaskItem["priority"], string> = {
    low: "#64748b",
    medium: "#0284c7",
    high: "#ea580c",
    critical: "#dc2626",
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 16px",
        marginBottom: "8px",
        backgroundColor: "#1e293b",
        borderRadius: "8px",
        borderLeft: `4px solid ${priorityColors[task.priority]}`,
      }}
    >
      <div style={{ flex: 1, marginRight: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 700,
              textTransform: "uppercase",
              padding: "2px 8px",
              borderRadius: "4px",
              backgroundColor: priorityColors[task.priority],
              color: "#ffffff",
            }}
          >
            {task.priority}
          </span>
          <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>{task.title}</span>
        </div>
        <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: "4px" }}>
          ID: <code>{task.id}</code> | Assigned: <strong>{task.assigned_worker}</strong> | Battery:{" "}
          {task.battery_level}%
        </div>
        <div style={{ marginTop: "8px" }}>
          <input
            type="text"
            value={operatorNote}
            onChange={(e) => setOperatorNote(e.target.value)}
            placeholder="Type notes (e.g. 'Rotor inspected') then re-order..."
            style={{
              width: "100%",
              maxWidth: "400px",
              padding: "6px 10px",
              fontSize: "0.85rem",
              borderRadius: "4px",
              border: "1px solid #475569",
              backgroundColor: "#0f172a",
              color: "#f1f5f9",
            }}
          />
        </div>
      </div>

      <div style={{ display: "flex", gap: "6px" }}>
        <button
          disabled={index === 0}
          onClick={() => onMove(index, "up")}
          style={{
            padding: "6px 12px",
            backgroundColor: index === 0 ? "#334155" : "#3b82f6",
            color: "#ffffff",
            border: "none",
            borderRadius: "4px",
            cursor: index === 0 ? "not-allowed" : "pointer",
          }}
        >
          ▲ Up
        </button>
        <button
          disabled={index === total - 1}
          onClick={() => onMove(index, "down")}
          style={{
            padding: "6px 12px",
            backgroundColor: index === total - 1 ? "#334155" : "#3b82f6",
            color: "#ffffff",
            border: "none",
            borderRadius: "4px",
            cursor: index === total - 1 ? "not-allowed" : "pointer",
          }}
        >
          ▼ Down
        </button>
      </div>
    </div>
  );
}

export const TaskQueueList: React.FC<TaskQueueListProps> = ({ tasks, onTasksChange }) => {
  /**
   * FLAGGED BEHAVIOR (Exercise 1):
   * 1. In-place array mutation: mutates the array directly and calls state setter,
   *    violating immutability and breaking Object.is equality checks.
   * 2. Index keys in .map((task, index) => ... key={index}): Fiber attaches the child
   *    Fiber node and its local state (like operatorNote) to the index position. When
   *    re-ordering, the text typed into Task A stays at index 0 and bleeds into Task B!
   */
  const handleMove = (index: number, direction: "up" | "down") => {
    const targetIndex = direction === "up" ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= tasks.length) return;

    // TODO(human): Exercise 1
    // 1. Avoid mutating the existing `tasks` array in place. Create a new copy
    //    using immutability patterns (e.g. `[...tasks]` or `tasks.slice()`).
    // 2. Swap elements cleanly within the cloned array and call `onTasksChange(cloned)`.
    // 3. In the JSX below, replace `key={index}` with a stable entity identifier (`key={task.id}`).
    //
    // Notice what happens BEFORE the fix:
    // Type a note in item 0, click "Down". The text remains in slot 0 instead of moving with the item!

    // --- BUGGY IMPLEMENTATION FOR EXPERIMENTATION ---
   // const buggyList = tasks; // Direct reference mutation!
   // const temp = buggyList[index];
   // buggyList[index] = buggyList[targetIndex];
   // buggyList[targetIndex] = temp;
   // onTasksChange([...buggyList]); // Force update but with mutated source array
    // -----------------------------------------------

    const new_tasts = [...tasks]
    const temp = new_tasts[index];
    new_tasts[index] = new_tasts[targetIndex];
    new_tasts[targetIndex] = temp;
    onTasksChange(new_tasts); 
    
  };

  return (
    <div style={{ backgroundColor: "#0f172a", padding: "16px", borderRadius: "8px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h3 style={{ margin: 0, color: "#f8fafc" }}>Fleet Dispatch Priority Queue</h3>
          <p style={{ margin: "4px 0 0 0", fontSize: "0.85rem", color: "#94a3b8" }}>
            Re-order tasks to test Fiber reconciliation. Observe whether operator notes remain bound to their task.
          </p>
        </div>
        <span style={{ fontSize: "0.85rem", color: "#38bdf8", fontWeight: 600 }}>
          {tasks.length} active tasks
        </span>
      </div>

      <div>
        {/*
          TODO(human): Exercise 1
          BUGGY KEY: key={index}
          Change key={index} to key={task.id} once you observe the state bleeding bug!
        */}
        {tasks.map((task, index) => (
          <TaskQueueItem
            key={task.id} // <-- BUG: Using array index as Fiber key!
            task={task}
            index={index}
            total={tasks.length}
            onMove={handleMove}
          />
        ))}
      </div>
    </div>
  );
};
