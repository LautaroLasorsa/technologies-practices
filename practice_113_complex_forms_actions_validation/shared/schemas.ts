import { z } from 'zod';

/**
 * ============================================================================
 * SHARED SCHEMAS: DOMAIN INVARIANTS & STRUCTURAL VALIDATION
 * ============================================================================
 * This module is imported by both the Frontend (React Hook Form + @hookform/resolvers/zod)
 * and the Backend (Fastify route parser).
 *
 * Sharing this single source of truth guarantees:
 * 1. Zero validation drift between client instant UI feedback and server enforcement.
 * 2. Strict typing across the wire for requests and responses.
 */

export const ShiftTypeEnum = z.enum(['standard', 'overnight'], {
  errorMap: () => ({ message: 'Shift type must be either standard or overnight' }),
});
export type ShiftType = z.infer<typeof ShiftTypeEnum>;

export const DepartmentEnum = z.enum(['assembly', 'machining', 'packaging', 'quality_control'], {
  errorMap: () => ({ message: 'Please select a valid production department' }),
});
export type Department = z.infer<typeof DepartmentEnum>;

export const ResourceRoleEnum = z.enum(['primary_operator', 'secondary_support', 'supervisor'], {
  errorMap: () => ({ message: 'Please select a valid resource role' }),
});
export type ResourceRole = z.infer<typeof ResourceRoleEnum>;

/**
 * Time regex format: "HH:MM" in 24-hour notation (e.g. "08:30", "17:00")
 */
export const TIME_REGEX = /^([01]\d|2[0-3]):([0-5]\d)$/;

/**
 * Helper to convert "HH:MM" string to minutes since midnight for invariant comparison
 */
export function timeStringToMinutes(timeStr: string): number {
  const [hours, minutes] = timeStr.split(':').map(Number);
  return hours * 60 + minutes;
}

/**
 * Schema for an individual resource allocation row inside the dynamic array
 */
export const resourceAllocationSchema = z.object({
  resourceId: z
    .string()
    .min(1, 'Resource ID is required')
    .regex(/^RES-[A-Z0-9-]+$/, 'Format must match RES-XXXX (e.g. RES-ROBOT-01, RES-LATHE-02)'),
  role: ResourceRoleEnum,
  allocatedHours: z
    .number({ invalid_type_error: 'Allocated hours must be a number' })
    .min(0.5, 'Minimum allocation is 0.5 hours')
    .max(12, 'Maximum allocation is 12 hours'),
  loadPercentage: z
    .number({ invalid_type_error: 'Load percentage must be a number' })
    .min(1, 'Minimum load is 1%')
    .max(100, 'Maximum load is 100%'),
});

export type ResourceAllocation = z.infer<typeof resourceAllocationSchema>;

/**
 * Base fields shared across all shift wizard steps
 */
const baseShiftSchema = z.object({
  shiftCode: z
    .string()
    .min(1, 'Shift code is required')
    .regex(/^SHIFT-\d{4}-\d{3}$/, 'Format must be SHIFT-YYYY-NNN (e.g. SHIFT-2026-001)'),
  department: DepartmentEnum,
  shiftType: ShiftTypeEnum,
  startTime: z
    .string()
    .min(1, 'Start time is required')
    .regex(TIME_REGEX, 'Start time must be in HH:MM 24-hour format (e.g. 08:00)'),
  endTime: z
    .string()
    .min(1, 'End time is required')
    .regex(TIME_REGEX, 'End time must be in HH:MM 24-hour format (e.g. 16:30)'),
  handoverSupervisor: z.string().optional(),
  notes: z.string().max(300, 'Notes cannot exceed 300 characters').optional(),
  allocations: z.array(resourceAllocationSchema),
});

/**
 * Complete Shift Wizard Schema with cross-field and array invariants via .superRefine()
 */
export const shiftWizardSchema = baseShiftSchema.superRefine((data, ctx) => {
  // --------------------------------------------------------------------------
  // TODO(human) [Exercise 1]: Cross-Field & Array Invariants Validation
  // --------------------------------------------------------------------------
  // Learning Objectives:
  // 1. Enforce cross-field chronological invariants:
  //    - If data.shiftType === 'standard':
  //      Convert startTime and endTime to minutes.
  //      If endTimeMinutes <= startTimeMinutes:
  //        Add an issue on path: ['endTime'] with message:
  //        "Standard shifts must end after start time on the same calendar day."
  //    - If data.shiftType === 'overnight':
  //      If !data.handoverSupervisor or data.handoverSupervisor.trim().length < 3:
  //        Add an issue on path: ['handoverSupervisor'] with message:
  //        "Overnight shifts require a designated handover supervisor (min 3 chars)."
  //      Also verify overnight spans past midnight:
  //      If endTimeMinutes >= startTimeMinutes:
  //        Add an issue on path: ['endTime'] with message:
  //        "Overnight shifts must cross midnight (end time must be numerically earlier than start time)."
  //
  // 2. Enforce array cardinality invariant:
  //    - If data.allocations.length === 0:
  //      Add an issue on path: ['allocations'] with message:
  //      "At least one resource must be allocated to the shift."
  //
  // 3. Enforce array item uniqueness invariant without full-form re-renders:
  //    - Track seen resourceId strings in a Set or Map.
  //    - For any index i where data.allocations[i].resourceId is repeated:
  //      Add an issue targeted specifically at path: ['allocations', i, 'resourceId']
  //      with message: `Duplicate resource '${data.allocations[i].resourceId}' in allocation list.`
  //
  // Hint: Use ctx.addIssue({ code: z.ZodIssueCode.custom, message, path })
  // --------------------------------------------------------------------------

  // Baseline implementation provided to enable type inference & compilation.
  // Replace or enhance with your complete invariant checks:
  if (data.allocations.length === 0) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'At least one resource must be allocated to the shift.',
      path: ['allocations'],
    });
  }

  // TODO: Add standard vs overnight time comparisons and duplicate resource checks here.
});

export type ShiftWizardInput = z.infer<typeof shiftWizardSchema>;

/**
 * API Response Payloads
 */
export interface ServerValidationErrorResponse {
  success: false;
  message: string;
  fieldErrors?: Record<string, string>;
}

export interface ShiftSubmitSuccessResponse {
  success: true;
  message: string;
  shiftId: string;
  data: ShiftWizardInput;
}

export interface QuickApprovalRequest {
  shiftId: string;
  supervisorNotes: string;
  overrideCode?: string;
}

export interface QuickApprovalResponse {
  success: boolean;
  message: string;
  timestamp: string;
  approvedBy: string;
  shiftId: string;
}
