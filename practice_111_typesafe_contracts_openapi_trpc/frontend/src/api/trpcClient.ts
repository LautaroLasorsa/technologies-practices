import { createTRPCClient, httpBatchLink } from "@trpc/client";
import type { AppRouter } from "../../../services/node-trpc/src/router.js";

/**
 * End-to-end type-safe tRPC client.
 *
 * Notice the import above: `import type { AppRouter }`!
 * This is a TYPE-ONLY import. When Vite / TypeScript builds this file,
 * the entire import is completely erased from JavaScript output.
 *
 * Zero server code, zero Zod schemas, zero Node dependencies are bundled
 * into the frontend. The client achieves 100% type safety via generic inference alone.
 */
export const trpc = createTRPCClient<AppRouter>({
  links: [
    httpBatchLink({
      url: "http://localhost:4111",
    }),
  ],
});
