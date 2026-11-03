/**
 * Exercise 1: Cancelable Fetch SSE Stream Parser
 *
 * Implements a streaming HTTP client for Server-Sent Events over POST
 * using fetch(), ReadableStream (response.body.getReader()), and TextDecoder.
 */

import { useState, useRef, useCallback } from 'react';
import type { SseRawEvent } from '../types';

export interface UseFetchSseStreamOptions {
  onEvent?: (event: SseRawEvent) => void;
  onToken?: (token: string) => void;
  onError?: (error: Error) => void;
  onDone?: () => void;
}

export interface UseFetchSseStreamReturn {
  startStream: (url: string, body: unknown) => Promise<void>;
  abortStream: () => void;
  tokens: string[];
  accumulatedText: string;
  isStreaming: boolean;
  eventCount: number;
  error: Error | null;
}

/**
 * Parses raw SSE string lines into structured SseRawEvent objects.
 * Format specification (W3C text/event-stream):
 *   id: <string>
 *   event: <string>
 *   data: <string>
 *   : <comment / heartbeat>
 * Empty line (\n\n) delimits consecutive events.
 */
export function parseSseBlock(block: string): SseRawEvent | null {
  const lines = block.split('\n');
  let eventType = 'message';
  let dataLines: string[] = [];
  let id: string | undefined = undefined;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith(':')) {
      // Empty line or comment ping
      continue;
    }
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    } else if (line.startsWith('id:')) {
      id = line.slice(3).trim();
    }
  }

  if (dataLines.length === 0 && eventType === 'message' && !id) {
    return null;
  }

  return {
    id,
    event: eventType,
    data: dataLines.join('\n'),
  };
}

export function useFetchSseStream(options: UseFetchSseStreamOptions = {}): UseFetchSseStreamReturn {
  const [tokens, setTokens] = useState<string[]>([]);
  const [accumulatedText, setAccumulatedText] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [eventCount, setEventCount] = useState<number>(0);
  const [error, setError] = useState<Error | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const abortStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  }, []);

  const startStream = useCallback(async (url: string, body: unknown) => {
    // Abort any existing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setIsStreaming(true);
    setError(null);
    setTokens([]);
    setAccumulatedText('');
    setEventCount(0);

    try {
      // --- TODO(human): Exercise 1 ---
      // Task:
      // 1. Issue an HTTP POST fetch request with headers:
      //      'Content-Type': 'application/json',
      //      'Accept': 'text/event-stream'
      //    and pass `signal: abortController.signal`.
      // 2. Validate `response.ok` and `response.body`. If not ok, throw new Error(response.statusText).
      // 3. Obtain reader: `const reader = response.body.getReader()`.
      // 4. Instantiate `const decoder = new TextDecoder('utf-8')`.
      // 5. In a while(true) loop:
      //    a. `const { done, value } = await reader.read();`
      //    b. If done, break.
      //    c. Decode the Uint8Array: `const chunk = decoder.decode(value, { stream: true });`
      //    d. Handle fragmented chunks: maintain an internal `buffer` string.
      //       Concatenate `buffer += chunk`. Split by double-newline `\n\n`.
      //       The last slice of the split is incomplete and remains in `buffer`.
      //       Complete blocks are parsed using `parseSseBlock(block)`.
      //    e. Extract tokens from data JSON and update state + call `options.onToken?.(token)`.
      // 6. Handle AbortError cleanly without setting error state.
      //
      // Starter Implementation:
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(body),
        signal: abortController.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`SSE request failed: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let localEventCount = 0;
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode incoming binary chunk, preserving incomplete multi-byte sequences
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // SSE events are delimited by double newline (\n\n or \r\n\r\n)
        const parts = buffer.split(/\r?\n\r?\n/);
        // The last element is either empty (if ended on boundary) or an incomplete event fragment
        buffer = parts.pop() || '';

        for (const block of parts) {
          if (!block.trim()) continue;

          const parsed = parseSseBlock(block);
          if (!parsed) continue;

          localEventCount++;
          options.onEvent?.(parsed);

          if (parsed.event === 'token') {
            try {
              const payload = JSON.parse(parsed.data);
              const tokenStr: string = payload.token || '';
              fullText += tokenStr;
              setTokens(prev => [...prev, tokenStr]);
              setAccumulatedText(fullText);
              options.onToken?.(tokenStr);
            } catch {
              // Plain string data fallback
              fullText += parsed.data;
              setTokens(prev => [...prev, parsed.data]);
              setAccumulatedText(fullText);
            }
          } else if (parsed.event === 'done') {
            options.onDone?.();
          }
          setEventCount(localEventCount);
        }
      }

      // Flush any residual complete event left in the buffer
      if (buffer.trim()) {
        const lastParsed = parseSseBlock(buffer);
        if (lastParsed) {
          options.onEvent?.(lastParsed);
        }
      }

      options.onDone?.();
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        console.log('[useFetchSseStream] Stream aborted by client.');
      } else {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        setError(errorObj);
        options.onError?.(errorObj);
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [options]);

  return {
    startStream,
    abortStream,
    tokens,
    accumulatedText,
    isStreaming,
    eventCount,
    error,
  };
}
