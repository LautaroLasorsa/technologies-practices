import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect } from 'vitest';
import { http, HttpResponse, delay } from 'msw';
import { server } from '../mocks/server';
import { LogInspector } from '../components/LogInspector';

describe('Network Boundary Race Condition & Cancellation Tests', () => {
  // =========================================================================
  // TODO(human): Exercise 4 - Deterministic Out-of-Order Network Race Test
  // =========================================================================
  // Instructions:
  // 1. Configure MSW with asymmetric delays:
  //    - 'warehouse-us-east' request is delayed by 150ms.
  //    - 'warehouse-eu-central' request is delayed by only 10ms.
  // 2. Simulate rapid partition switching via partition-select.
  // 3. Assert that when the slow, earlier request finally resolves, it does NOT
  //    overwrite the data from the newer, faster request.

  it('prevents stale out-of-order responses from overwriting newer partition state', async () => {
    // Intercept GET /api/logs with custom delays per partition
    server.use(
      http.get('*/api/logs', async ({ request }) => {
        const url = new URL(request.url);
        const partition = url.searchParams.get('partition');

        if (partition === 'warehouse-us-east') {
          // Slow response: 150ms delay
          await delay(150);
          return HttpResponse.json({
            partition: 'warehouse-us-east',
            total: 100,
            logs: [
              {
                id: 'log-warehouse-us-east-stale',
                timestamp: new Date().toISOString(),
                level: 'WARN',
                service: 'scheduler-engine',
                message: 'STALE LOG FROM US-EAST',
                latencyMs: 120,
                anomalyScore: 0.1,
                partition: 'warehouse-us-east',
              },
            ],
          });
        }

        if (partition === 'warehouse-eu-central') {
          // Fast response: 10ms delay
          await delay(10);
          return HttpResponse.json({
            partition: 'warehouse-eu-central',
            total: 100,
            logs: [
              {
                id: 'log-warehouse-eu-central-fresh',
                timestamp: new Date().toISOString(),
                level: 'INFO',
                service: 'order-router',
                message: 'FRESH LOG FROM EU-CENTRAL',
                latencyMs: 15,
                anomalyScore: 0.05,
                partition: 'warehouse-eu-central',
              },
            ],
          });
        }

        return HttpResponse.json({ partition, total: 0, logs: [] });
      })
    );

    const user = userEvent.setup();
    render(<LogInspector initialPartition="warehouse-us-east" />);

    const select = screen.getByTestId('partition-select');

    // Immediately switch to warehouse-eu-central before the 150ms US-EAST response finishes
    await user.selectOptions(select, 'warehouse-eu-central');

    // Verify EU-CENTRAL (the newer request) arrives and renders
    await waitFor(
      () => {
        expect(
          screen.getByTestId('log-row-log-warehouse-eu-central-fresh')
        ).toBeInTheDocument();
      },
      { timeout: 1000 }
    );

    // Wait 250ms for the delayed US-EAST network promise to resolve in the background
    await new Promise((resolve) => setTimeout(resolve, 250));

    // Stale US-EAST log MUST NOT have overwritten EU-CENTRAL
    expect(
      screen.queryByTestId('log-row-log-warehouse-us-east-stale')
    ).not.toBeInTheDocument();

    expect(
      screen.getByTestId('log-row-log-warehouse-eu-central-fresh')
    ).toBeInTheDocument();
  });
});
