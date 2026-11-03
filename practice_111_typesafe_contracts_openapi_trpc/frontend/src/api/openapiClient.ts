import createClient from "openapi-fetch";
import type { paths, components } from "./schema.js";

export type OptimizationRequest = components["schemas"]["OptimizationRequest"];
export type OptimizationSuccessResponse = components["schemas"]["OptimizationSuccessResponse"];
export type OptimizationConflictError = components["schemas"]["OptimizationConflictError"];
export type OptimizationResourceExhaustedError =
  components["schemas"]["OptimizationResourceExhaustedError"];

// Discriminated result union for the frontend UI
export type OptimizationApiResult =
  | {
      kind: "SUCCESS";
      data: OptimizationSuccessResponse;
    }
  | {
      kind: "CONFLICT";
      error: OptimizationConflictError;
    }
  | {
      kind: "EXHAUSTED";
      error: OptimizationResourceExhaustedError;
    }
  | {
      kind: "UNKNOWN_ERROR";
      statusCode: number;
      message: string;
    };

/**
 * Type-safe HTTP Client powered by openapi-fetch.
 * Zero-runtime schema overhead: paths, parameters, and payloads are enforced
 * at compile time via TypeScript's indexed access types.
 */
export const openapiClient = createClient<paths>({
  baseUrl: "http://localhost:8111",
});

// TODO(human): Exercise 1
// Complete this typed client wrapper using `openapi-fetch`.
// Notice how `client.POST` auto-completes available endpoints and expects
// the exact `requestBody` typed in Python Pydantic.
// Use pattern-matching on HTTP status codes to produce the discriminated union `OptimizationApiResult`.
export async function triggerOptimization(
  params: OptimizationRequest
): Promise<OptimizationApiResult> {
  const { data, error, response } = await openapiClient.POST("/api/v1/optimize", {
    body: params,
  });

  if (response.status === 200 && data) {
    return {
      kind: "SUCCESS",
      data,
    };
  }

  if (response.status === 409 && error) {
    // Type narrowing: error is statically typed as OptimizationConflictError for status 409!
    return {
      kind: "CONFLICT",
      error: error as OptimizationConflictError,
    };
  }

  if (response.status === 422 && error) {
    return {
      kind: "EXHAUSTED",
      error: error as OptimizationResourceExhaustedError,
    };
  }

  return {
    kind: "UNKNOWN_ERROR",
    statusCode: response.status,
    message: response.statusText || "An unexpected error occurred during optimization request",
  };
}

export async function fetchOptimizationJob(jobId: string) {
  const { data, response } = await openapiClient.GET("/api/v1/jobs/{job_id}", {
    params: {
      path: {
        job_id: jobId,
      },
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch job ${jobId}: ${response.statusText}`);
  }

  return data;
}
