import { initTRPC, TRPCError } from "@trpc/server";
import type { CreateHTTPContextOptions } from "@trpc/server/adapters/standalone";

/**
 * Context creation per request.
 * Contains request metadata, correlation ID, and timestamp.
 */
export async function createContext(opts?: CreateHTTPContextOptions) {
  const userAgent = opts?.req.headers["user-agent"] ?? "unknown";
  const requestId = `req-${Math.random().toString(36).substring(2, 9)}`;

  return {
    requestId,
    userAgent,
    timestamp: new Date().toISOString(),
  };
}

export type Context = Awaited<ReturnType<typeof createContext>>;

const t = initTRPC.context<Context>().create();

/**
 * Middleware: logs request lifecycle and execution latency.
 */
const loggerMiddleware = t.middleware(async ({ path, type, next }) => {
  const start = performance.now();
  const result = await next();
  const durationMs = Math.round(performance.now() - start);
  console.log(`[tRPC] ${type} ${path} - ${durationMs}ms`);
  return result;
});

export const router = t.router;
export const publicProcedure = t.procedure.use(loggerMiddleware);
export { TRPCError };
