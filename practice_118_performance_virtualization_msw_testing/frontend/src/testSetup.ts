import '@testing-library/jest-dom';
import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './mocks/server';

beforeAll(() => {
  // Intercept all requests at the network boundary; fail fast on unhandled routes
  server.listen({ onUnhandledRequest: 'error' });
});

afterEach(() => {
  // Reset any runtime request handlers added during individual tests
  server.resetHandlers();
});

afterAll(() => {
  // Clean up and restore native network behavior
  server.close();
});
