"use client";

import { useTransition, useState } from "react";
import { createScenarioAction } from "@/app/actions/scenarioActions";

export default function ScenarioForm() {
  const [isPending, startTransition] = useTransition();
  const [feedback, setFeedback] = useState<{ success: boolean; message: string } | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFeedback(null);
    const form = event.currentTarget;
    const formData = new FormData(form);

    startTransition(async () => {
      try {
        const result = await createScenarioAction(formData);
        if (result.success) {
          setFeedback({ success: true, message: `Scenario created successfully: ${result.scenario?.name}` });
          form.reset();
        } else {
          setFeedback({ success: false, message: result.error || "Failed to create scenario" });
        }
      } catch (err: any) {
        setFeedback({ success: false, message: err.message || "An unexpected error occurred" });
      }
    });
  }

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: "600px" }}>
      <div className="form-group">
        <label className="form-label" htmlFor="name">
          Scenario Name
        </label>
        <input
          id="name"
          name="name"
          type="text"
          required
          placeholder="e.g. Black Friday Ingress Surge"
          className="form-input"
          minLength={3}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        <div className="form-group">
          <label className="form-label" htmlFor="target_growth_pct">
            Target Growth (%)
          </label>
          <input
            id="target_growth_pct"
            name="target_growth_pct"
            type="number"
            step="0.1"
            min="1"
            max="500"
            required
            defaultValue="25.0"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="allocated_budget_usd">
            Allocated Budget (USD)
          </label>
          <input
            id="allocated_budget_usd"
            name="allocated_budget_usd"
            type="number"
            min="100"
            step="100"
            required
            defaultValue="12000"
            className="form-input"
          />
        </div>
      </div>

      <div className="form-group">
        <label className="form-label" htmlFor="notes">
          Simulation Notes / Provisioning Strategy
        </label>
        <textarea
          id="notes"
          name="notes"
          rows={3}
          placeholder="Details on worker instance types, cache sharding, or regional placement..."
          className="form-input"
          style={{ resize: "vertical" }}
        />
      </div>

      {feedback && (
        <div
          className="callout"
          style={{
            borderLeftColor: feedback.success ? "var(--accent-green)" : "var(--accent-rose)",
            background: feedback.success ? "rgba(16, 185, 129, 0.1)" : "rgba(244, 63, 94, 0.1)",
            color: feedback.success ? "#34d399" : "#fb7185",
            marginBottom: "1rem",
          }}
        >
          {feedback.message}
        </div>
      )}

      <button type="submit" disabled={isPending} className="btn btn-primary">
        {isPending ? "Submitting to Internal Python Microservice..." : "Register Scenario (Server Action)"}
      </button>
    </form>
  );
}
