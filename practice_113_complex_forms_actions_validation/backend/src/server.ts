import fastify from 'fastify';
import cors from '@fastify/cors';
import {
  shiftWizardSchema,
  timeStringToMinutes,
  type ShiftWizardInput,
  type ServerValidationErrorResponse,
  type ShiftSubmitSuccessResponse,
  type QuickApprovalResponse,
} from '../../shared/schemas.js';

const app = fastify({ logger: true });

await app.register(cors, {
  origin: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
});

const PORT = 4113;

/**
 * ============================================================================
 * STATEFUL PERSISTENCE & RESOURCE RESERVATION MOCK (DATABASE SIMULATION)
 * ============================================================================
 * These represent database records that CANNOT be known to the frontend schema.
 * Only the backend can execute queries to detect collisions against this state.
 */
interface BookedWindow {
  resourceId: string;
  shiftCode: string;
  startTime: string;
  endTime: string;
}

const ACTIVE_SHIFT_CODES = new Set<string>([
  'SHIFT-2026-001',
  'SHIFT-2026-002',
]);

const BOOKED_RESOURCES: BookedWindow[] = [
  {
    resourceId: 'RES-ROBOT-01',
    shiftCode: 'SHIFT-2026-001',
    startTime: '08:00',
    endTime: '16:00',
  },
  {
    resourceId: 'RES-LATHE-02',
    shiftCode: 'SHIFT-2026-002',
    startTime: '12:00',
    endTime: '20:00',
  },
];

/**
 * Healthcheck route
 */
app.get('/health', async () => {
  return { status: 'ok', timestamp: new Date().toISOString() };
});

/**
 * Inspection route to query existing database reservations
 */
app.get('/api/shifts/booked-windows', async () => {
  return {
    activeShiftCodes: Array.from(ACTIVE_SHIFT_CODES),
    bookedResources: BOOKED_RESOURCES,
  };
});

/**
 * Shift Wizard Submission Route
 * Combines Layer 1 (Shared Zod Invariant Parsing) and Layer 2 (Asymmetric Business Rules)
 */
app.post<{
  Body: unknown;
  Reply: ShiftSubmitSuccessResponse | ServerValidationErrorResponse;
}>('/api/shifts', async (request, reply) => {
  // --------------------------------------------------------------------------
  // LAYER 1: Symmetric Structural & Invariant Validation (Shared Zod)
  // --------------------------------------------------------------------------
  const parseResult = shiftWizardSchema.safeParse(request.body);
  if (!parseResult.success) {
    const fieldErrors: Record<string, string> = {};
    for (const issue of parseResult.error.issues) {
      const path = issue.path.join('.');
      if (!fieldErrors[path]) {
        fieldErrors[path] = issue.message;
      }
    }
    return reply.status(400).send({
      success: false,
      message: 'Schema structural validation failed (Layer 1)',
      fieldErrors,
    });
  }

  const shiftData: ShiftWizardInput = parseResult.data;
  const businessFieldErrors: Record<string, string> = {};

  // --------------------------------------------------------------------------
  // TODO(human) [Exercise 3: Backend Stateful Business Validation]
  // --------------------------------------------------------------------------
  // Learning Objectives:
  // 1. Asymmetric Unique Invariant Check:
  //    - Check if ACTIVE_SHIFT_CODES.has(shiftData.shiftCode).
  //    - If true, populate:
  //      businessFieldErrors['shiftCode'] =
  //        `Shift code '${shiftData.shiftCode}' is already registered in the production database.`;
  //
  // 2. Asymmetric Resource Schedule Collision Check:
  //    - For each allocation in shiftData.allocations at index `i`:
  //      Check against BOOKED_RESOURCES for any entry where
  //      booked.resourceId === allocation.resourceId.
  //      Calculate time overlap between [shiftData.startTime, shiftData.endTime]
  //      and [booked.startTime, booked.endTime] using timeStringToMinutes().
  //      Two standard intervals [s1, e1] and [s2, e2] overlap if:
  //        Math.max(s1, s2) < Math.min(e1, e2)
  //      If an overlap occurs, attach a field-specific error:
  //        businessFieldErrors[`allocations.${i}.resourceId`] =
  //          `Resource '${allocation.resourceId}' is already booked during ${booked.startTime}-${booked.endTime} by shift ${booked.shiftCode}.`;
  //
  // 3. Return 422 Unprocessable Entity if businessFieldErrors has any keys:
  //    if (Object.keys(businessFieldErrors).length > 0) {
  //      return reply.status(422).send({
  //        success: false,
  //        message: 'Business rule collision detected (Layer 2)',
  //        fieldErrors: businessFieldErrors,
  //      });
  //    }
  // --------------------------------------------------------------------------

  // Baseline mock check (to be replaced/extended by Exercise 3):
  if (ACTIVE_SHIFT_CODES.has(shiftData.shiftCode)) {
    businessFieldErrors['shiftCode'] = `Shift code '${shiftData.shiftCode}' is already registered in the production database.`;
  }

  if (Object.keys(businessFieldErrors).length > 0) {
    return reply.status(422).send({
      success: false,
      message: 'Business rule validation failed (Layer 2)',
      fieldErrors: businessFieldErrors,
    });
  }

  // Persist new shift into database
  ACTIVE_SHIFT_CODES.add(shiftData.shiftCode);
  for (const alloc of shiftData.allocations) {
    BOOKED_RESOURCES.push({
      resourceId: alloc.resourceId,
      shiftCode: shiftData.shiftCode,
      startTime: shiftData.startTime,
      endTime: shiftData.endTime,
    });
  }

  const generatedId = `shift_${Date.now()}`;
  return reply.status(201).send({
    success: true,
    message: 'Shift created and resources locked successfully',
    shiftId: generatedId,
    data: shiftData,
  });
});

/**
 * Atomic Shift Quick-Approval Action Endpoint (Used by Exercise 4: React 19 Action)
 */
app.post<{
  Params: { id: string };
  Body: { supervisorNotes?: string; overrideCode?: string };
  Reply: QuickApprovalResponse | { success: false; message: string };
}>('/api/shifts/:id/quick-approval', async (request, reply) => {
  const { id } = request.params;
  const { supervisorNotes, overrideCode } = request.body || {};

  // Artificial latency to clearly visualize React 19 transition / useFormStatus pending states
  await new Promise((resolve) => setTimeout(resolve, 800));

  if (!supervisorNotes || supervisorNotes.trim().length < 5) {
    return reply.status(400).send({
      success: false,
      message: 'Supervisor sign-off notes must be at least 5 characters.',
    });
  }

  return reply.send({
    success: true,
    message: overrideCode
      ? `Shift ${id} approved with manager emergency override code [${overrideCode}].`
      : `Shift ${id} approved for production line execution.`,
    timestamp: new Date().toISOString(),
    approvedBy: 'Lead Supervisor (Station 4)',
    shiftId: id,
  });
});

const start = async () => {
  try {
    await app.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`\n======================================================`);
    console.log(`🚀 Fastify Backend listening on http://localhost:${PORT}`);
    console.log(`📡 Endpoints:`);
    console.log(`   - GET  /health`);
    console.log(`   - GET  /api/shifts/booked-windows`);
    console.log(`   - POST /api/shifts`);
    console.log(`   - POST /api/shifts/:id/quick-approval`);
    console.log(`======================================================\n`);
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
};

start();
