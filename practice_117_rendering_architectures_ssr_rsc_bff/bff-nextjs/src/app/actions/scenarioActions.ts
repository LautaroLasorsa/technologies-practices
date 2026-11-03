"use server";

import { revalidatePath, revalidateTag } from "next/cache";
import { Scenario } from "@/lib/api";

export interface CreateScenarioResult {
  success: boolean;
  scenario?: Scenario;
  error?: string;
}

/**
 * EXERCISE 3: Authenticated Server Action Mutation
 *
 * THE PROBLEM:
 * The browser cannot and should not communicate directly with the internal Python microservice.
 * The Python endpoint requires `X-Internal-Service-Key: super-secret-internal-key`, which would
 * be compromised if exposed to the client.
 *
 * THE SOLUTION:
 * In Next.js App Router, functions marked with `'use server'` become RPC endpoints callable
 * directly from client components or form submissions.
 * They execute strictly in the Node.js server environment, where `process.env.INTERNAL_SERVICE_KEY`
 * remains safe. After performing the mutation on the internal service, the action triggers
 * instant cache revalidation (`revalidatePath` or `revalidateTag`) so React streams the updated UI.
 *
 * TODO(human) [Exercise 3]:
 * 1. Extract values from `formData` (`name`, `target_growth_pct`, `allocated_budget_usd`, `notes`).
 * 2. Perform server-side validation (e.g. check min length, positive numeric ranges).
 * 3. Make an authenticated `fetch` request to `${PYTHON_SERVICE_URL}/internal/scenarios`
 *    with header `X-Internal-Service-Key: process.env.INTERNAL_SERVICE_KEY`.
 * 4. Check response status. If failed, return `{ success: false, error: message }`.
 * 5. If successful, parse the created `Scenario` and call `revalidatePath('/dashboard')`
 *    and/or `revalidateTag('scenarios')` to update the server cache.
 * 6. Return `{ success: true, scenario: createdScenario }`.
 */
export async function createScenarioAction(formData: FormData): Promise<CreateScenarioResult> {
  const pythonUrl = process.env.PYTHON_SERVICE_URL || "http://127.0.0.1:8117";
  const internalKey = process.env.INTERNAL_SERVICE_KEY || "super-secret-internal-key";

  const name = formData.get("name")?.toString().trim();
  const targetGrowthStr = formData.get("target_growth_pct")?.toString();
  const budgetStr = formData.get("allocated_budget_usd")?.toString();
  const notes = formData.get("notes")?.toString().trim();

  // Basic validation check
  if (!name || name.length < 3) {
    return { success: false, error: "Scenario name must be at least 3 characters long." };
  }

  const target_growth_pct = parseFloat(targetGrowthStr || "0");
  const allocated_budget_usd = parseFloat(budgetStr || "0");

  if (isNaN(target_growth_pct) || target_growth_pct <= 0) {
    return { success: false, error: "Target growth percentage must be a positive number." };
  }

  if (isNaN(allocated_budget_usd) || allocated_budget_usd < 100) {
    return { success: false, error: "Allocated budget must be at least $100." };
  }

  try {
    // TODO(human) [Exercise 3]: Dispatch the authenticated POST request to the Python microservice
    const response = await fetch(`${pythonUrl}/internal/scenarios`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Service-Key": internalKey,
      },
      body: JSON.stringify({
        name,
        target_growth_pct,
        allocated_budget_usd,
        notes: notes || null,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      return {
        success: false,
        error: `Python microservice returned HTTP ${response.status}: ${errorBody}`,
      };
    }

    const createdScenario: Scenario = await response.json();

    // Revalidate dashboard so Server Components fetch and stream the updated scenario list
    revalidatePath("/dashboard");
    revalidateTag("scenarios");

    return {
      success: true,
      scenario: createdScenario,
    };
  } catch (err: any) {
    return {
      success: false,
      error: `Failed to connect to internal microservice: ${err.message}`,
    };
  }
}
