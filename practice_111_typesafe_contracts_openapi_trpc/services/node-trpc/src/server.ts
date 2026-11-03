import { createHTTPServer } from "@trpc/server/adapters/standalone";
import cors from "cors";
import { createContext } from "./trpc.js";
import { appRouter } from "./router.js";

const PORT = 4111;

const server = createHTTPServer({
  middleware: cors({
    origin: ["http://localhost:5173", "http://127.0.0.1:5173"],
    credentials: true,
  }),
  router: appRouter,
  createContext,
});

server.listen(PORT, () => {
  console.log(`[tRPC Server] Ready and listening at http://localhost:${PORT}`);
  console.log(`[tRPC Server] Health & endpoint: http://localhost:${PORT}/workspace.list`);
});
