import { z } from "zod";

/**
 * Zod Schema for defensive runtime validation at the network boundary.
 *
 * Why do we need this if we already have TypeScript types?
 * Because TypeScript types DO NOT EXIST at runtime.
 * If a third-party microservice, legacy backend, or version-lagged API
 * silently renames a field, TypeScript's `as OptimizationSuccessResponse`
 * or unvalidated `data` will blindly allow the object through until a runtime
 * property access crashes the browser.
 */
export const ExternalScheduleSlotSchema = z.object({
  slot_id: z.string().min(1, "slot_id cannot be empty"),
  worker_id: z.string().min(1, "worker_id cannot be empty"),
  shift_name: z.string(),
  day_offset: z.number().int().nonnegative(),
  estimated_cost: z.number().nonnegative(),
});

export const ExternalOptimizationPayloadSchema = z.object({
  status: z.enum(["COMPLETED", "QUEUED"]),
  job_id: z.string().min(1),
  horizon_days: z.number().int().positive(),
  total_cost: z.number().nonnegative(),
  solver_status: z.string(),
  assignments: z.array(ExternalScheduleSlotSchema),
  created_at: z.string().datetime({ message: "Must be a valid ISO-8601 datetime" }),
});

export type ExternalOptimizationPayload = z.infer<
  typeof ExternalOptimizationPayloadSchema
>;

export type BoundaryValidationResult =
  | {
      success: true;
      data: ExternalOptimizationPayload;
      validationDurationMs: number;
    }
  | {
      success: false;
      errorMessages: string[];
      rawPayloadSample: string;
      driftDetected: boolean;
    };

// TODO(human): Exercise 4
// Implement this runtime boundary guard.
// Use `ExternalOptimizationPayloadSchema.safeParse` to inspect untrusted raw data.
// In the event of a failure, format the Zod issues into clear diagnostics, identifying
// whether a missing or renamed field caused Schema Drift.
export function validateContractBoundary(
  rawPayload: unknown
): BoundaryValidationResult {
  const start = performance.now();
  const parseResult = ExternalOptimizationPayloadSchema.safeParse(rawPayload);
  const duration = Math.round((performance.now() - start) * 100) / 100;

  if (parseResult.success) {
    return {
      success: true,
      data: parseResult.data,
      validationDurationMs: duration,
    };
  }

  const issues = parseResult.error.issues;
  const errorMessages = issues.map((issue) => {
    const pathStr = issue.path.join(".") || "root";
    return `[Path: ${pathStr}] ${issue.message} (code: ${issue.code})`;
  });

  // Check if standard drift indicators exist (e.g. required field missing or wrong type)
  const driftDetected = issues.some(
    (issue) =>
      issue.code === "invalid_type" && issue.received === "undefined"
  );

  return {
    success: false,
    errorMessages,
    rawPayloadSample: JSON.stringify(rawPayload, null, 2),
    driftDetected,
  };
}
