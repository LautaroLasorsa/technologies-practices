import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { server } from '../mocks/server';
import { LogInspector } from '../components/LogInspector';

describe('LogInspector Integration Tests (Real Wire with MSW v2)', () => {
  // =========================================================================
  // TODO(human): Exercise 3 - Network Boundary Integration Testing with MSW v2
  // =========================================================================
  // Instructions:
  // Write integration tests that exercise the real network stack (Component -> fetch -> MSW).
  // Do NOT mock `global.fetch` or React hooks!

  it('fetches and renders initial dataset over the wire without mocked fetch', async () => {
    render(<LogInspector initialPartition="warehouse-us-east" />);

    // Verify initial network fetch completes and updates the performance HUD
    await waitFor(() => {
      expect(screen.getByTestId('performance-hud')).toHaveTextContent(/Loaded:\s*1,000/);
    });

    // Check that virtualized grid container is mounted
    const grid = screen.getByTestId('log-grid-container');
    expect(grid).toBeInTheDocument();

    // Verify at least one log row exists in DOM
    const row = await screen.findByTestId(/^log-row-log-warehouse-us-east/);
    expect(row).toBeInTheDocument();
  });

  it('filters dataset using client-side worker after wire payload arrival', async () => {
    const user = userEvent.setup();
    render(<LogInspector initialPartition="warehouse-us-east" />);

    // Wait for data load
    await waitFor(() => {
      expect(screen.getByTestId('performance-hud')).toHaveTextContent(/Loaded:\s*1,000/);
    });

    // Type a specific search query into the search input
    const searchInput = screen.getByTestId('search-input');
    await user.type(searchInput, 'scheduler-engine');

    // Wait for the worker computation to complete and HUD to reflect matched count
    await waitFor(() => {
      expect(screen.getByTestId('performance-hud')).toHaveTextContent(/Matched:\s*[1-9]/);
    });
  });

  it('handles HTTP 500 Internal Server Error and recovers via Retry button', async () => {
    let failureCount = 1;

    // Dynamically override MSW handler to fail once then succeed
    server.use(
      http.get('*/api/logs', () => {
        if (failureCount > 0) {
          failureCount--;
          return HttpResponse.json(
            { error: 'Postgres primary failover in progress' },
            { status: 500 }
          );
        }
        return HttpResponse.json({
          partition: 'warehouse-us-east',
          total: 1000,
          logs: [],
        });
      })
    );

    const user = userEvent.setup();
    render(<LogInspector initialPartition="warehouse-us-east" />);

    // Verify error banner is displayed with HTTP 500 details
    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent(/HTTP 500: Postgres primary failover in progress/i);

    // Click retry button
    const retryBtn = screen.getByTestId('retry-button');
    await user.click(retryBtn);

    // Verify banner disappears on successful recovery
    await waitFor(() => {
      expect(screen.queryByTestId('error-banner')).not.toBeInTheDocument();
    });
  });

  it('handles HTTP 429 Rate Limiting with Retry-After header', async () => {
    server.use(
      http.get('*/api/logs', () => {
        return HttpResponse.json(
          { error: 'Rate limit exceeded: quota depleted' },
          {
            status: 429,
            headers: {
              'Retry-After': '12',
              'Content-Type': 'application/json',
            },
          }
        );
      })
    );

    render(<LogInspector initialPartition="warehouse-us-east" />);

    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent(/HTTP 429: Rate limit exceeded\. Retry allowed in 12s/i);
  });
});
