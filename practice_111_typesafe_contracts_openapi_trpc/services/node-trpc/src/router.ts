import { z } from "zod";
import { publicProcedure, router, TRPCError } from "./trpc.js";

// Mock workspace database
interface WorkspaceRecord {
  id: string;
  name: string;
  tier: "STARTER" | "GROWTH" | "ENTERPRISE";
  maxOvertimeHours: number;
  allowWeekendShifts: boolean;
  updatedAt: string;
}

const WORKSPACE_DB: Record<string, WorkspaceRecord> = {
  "ws-alpha": {
    id: "ws-alpha",
    name: "Logistics Hub North",
    tier: "GROWTH",
    maxOvertimeHours: 20,
    allowWeekendShifts: true,
    updatedAt: new Date().toISOString(),
  },
  "ws-beta": {
    id: "ws-beta",
    name: "Urban Delivery Beta",
    tier: "STARTER",
    maxOvertimeHours: 8,
    allowWeekendShifts: false,
    updatedAt: new Date().toISOString(),
  },
};

/**
 * Workspace management router.
 * Demonstrates code-first type inference and compile-time contract sharing.
 */
const workspaceRouter = router({
  list: publicProcedure.query(async () => {
    return Object.values(WORKSPACE_DB);
  }),

  getById: publicProcedure
    .input(z.object({ workspaceId: z.string().min(1) }))
    .query(async ({ input }) => {
      const workspace = WORKSPACE_DB[input.workspaceId];
      if (!workspace) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: `Workspace '${input.workspaceId}' not found.`,
        });
      }
      return workspace;
    }),

  // TODO(human): Exercise 2
  // Implement this tRPC mutation with Zod input validation, context checking,
  // and domain-specific TRPCError error throwing.
  // Notice that NO codegen is required: modifying the Zod input or return shape
  // instantly updates `AppRouter` and type-checks the React frontend.
  updateSchedulePolicy: publicProcedure
    .input(
      z.object({
        workspaceId: z.string().min(1, "Workspace ID is required"),
        maxOvertimeHours: z
          .number()
          .min(0, "Overtime cannot be negative")
          .max(40, "Overtime cannot exceed 40 hours"),
        allowWeekendShifts: z.boolean(),
        tier: z.enum(["STARTER", "GROWTH", "ENTERPRISE"]),
      })
    )
    .mutation(async ({ input, ctx }) => {
      const current = WORKSPACE_DB[input.workspaceId];
      if (!current) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: `Workspace '${input.workspaceId}' does not exist.`,
        });
      }

      // Domain constraint: Starter tier cannot exceed 10 hours overtime
      if (input.tier === "STARTER" && input.maxOvertimeHours > 10) {
        throw new TRPCError({
          code: "CONFLICT",
          message:
            "Starter tier workspaces are capped at 10 hours overtime. Upgrade to GROWTH or reduce hours.",
        });
      }

      const updated: WorkspaceRecord = {
        ...current,
        tier: input.tier,
        maxOvertimeHours: input.maxOvertimeHours,
        allowWeekendShifts: input.allowWeekendShifts,
        updatedAt: new Date().toISOString(),
      };

      WORKSPACE_DB[input.workspaceId] = updated;

      return {
        success: true as const,
        workspace: updated,
        requestId: ctx.requestId,
      };
    }),
});

export const appRouter = router({
  workspace: workspaceRouter,
});

/**
 * Type-only export.
 * In a monorepo or full-stack setup, the frontend imports:
 * `import type { AppRouter } from '...'`
 * Zero runtime code leaks into the client bundle!
 */
export type AppRouter = typeof appRouter;
