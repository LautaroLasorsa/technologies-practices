/**
 * Exercise 2: UI Backpressure via Ring Buffer & useSyncExternalStore
 *
 * Decouples high-frequency network ingestion (250+ packets/sec) from the
 * browser's 60Hz render loop to eliminate UI stutter and main-thread starvation.
 */

import { useSyncExternalStore } from 'react';

/**
 * Fixed-capacity circular buffer with O(1) append and zero reallocation.
 */
export class RingBuffer<T> {
  private buffer: (T | undefined)[];
  private capacity: number;
  private head: number = 0; // Index of oldest element
  private tail: number = 0; // Index where next element will be written
  private size: number = 0;

  constructor(capacity: number) {
    if (capacity <= 0) throw new Error('RingBuffer capacity must be > 0');
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  /**
   * Appends an element in O(1). If full, overwrites the oldest element.
   */
  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;

    if (this.size < this.capacity) {
      this.size++;
    } else {
      // Overwriting oldest: advance head pointer
      this.head = (this.head + 1) % this.capacity;
    }
  }

  /**
   * Returns current number of elements stored.
   */
  get length(): number {
    return this.size;
  }

  /**
   * Clears the ring buffer.
   */
  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.size = 0;
    this.buffer.fill(undefined);
  }

  /**
   * Produces an ordered snapshot array from oldest to newest.
   */
  toArray(): T[] {
    const result: T[] = new Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const idx = (this.head + i) % this.capacity;
      result[i] = this.buffer[idx] as T;
    }
    return result;
  }

  /**
   * Returns the most recently appended item, or undefined if empty.
   */
  peekLatest(): T | undefined {
    if (this.size === 0) return undefined;
    const lastIdx = (this.tail - 1 + this.capacity) % this.capacity;
    return this.buffer[lastIdx];
  }
}

export interface RingBufferStore<T> {
  push: (item: T) => void;
  clear: () => void;
  subscribe: (onStoreChange: () => void) => () => void;
  getSnapshot: () => readonly T[];
  getStats: () => {
    totalIngested: number;
    bufferSize: number;
    capacity: number;
    flushCount: number;
  };
}

/**
 * Creates an external store conforming to React 19 useSyncExternalStore contract.
 * Coalesces rapid network pushes into a single requestAnimationFrame tick.
 */
export function createRingBufferStore<T>(capacity: number = 500): RingBufferStore<T> {
  const ringBuffer = new RingBuffer<T>(capacity);
  const listeners = new Set<() => void>();

  // Cached immutable snapshot reference required by useSyncExternalStore (Object.is)
  let cachedSnapshot: readonly T[] = Object.freeze([]);
  let isFlushScheduled = false;
  let rafId: number | null = null;

  let totalIngested = 0;
  let flushCount = 0;

  const flush = () => {
    isFlushScheduled = false;
    rafId = null;

    // --- TODO(human): Exercise 2 (Snapshot Update & Notification) ---
    // Task:
    // 1. Generate an immutable array from the mutable ring buffer:
    //      cachedSnapshot = Object.freeze(ringBuffer.toArray());
    // 2. Increment flushCount.
    // 3. Notify all subscribed React components by calling each listener in `listeners`.
    //
    cachedSnapshot = Object.freeze(ringBuffer.toArray());
    flushCount++;

    for (const listener of listeners) {
      try {
        listener();
      } catch (err) {
        console.error('[RingBufferStore] Listener error:', err);
      }
    }
  };

  const push = (item: T) => {
    // --- TODO(human): Exercise 2 (Push with Coalesced Backpressure) ---
    // Task:
    // 1. Append the item into the internal RingBuffer in O(1).
    // 2. Increment `totalIngested`.
    // 3. If a flush is NOT already scheduled (`!isFlushScheduled`):
    //    a. Set `isFlushScheduled = true`.
    //    b. Schedule a flush via `requestAnimationFrame(flush)`.
    //    c. Store the request ID in `rafId`.
    // Note: Multiple pushes arriving within the same 16ms animation frame
    // update the buffer immediately in-place, but trigger only ONE render pass!
    //
    ringBuffer.push(item);
    totalIngested++;

    if (!isFlushScheduled) {
      isFlushScheduled = true;
      rafId = requestAnimationFrame(flush);
    }
  };

  const clear = () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    isFlushScheduled = false;
    ringBuffer.clear();
    cachedSnapshot = Object.freeze([]);
    for (const listener of listeners) {
      listener();
    }
  };

  const subscribe = (onStoreChange: () => void) => {
    listeners.add(onStoreChange);
    return () => {
      listeners.delete(onStoreChange);
      if (listeners.size === 0 && rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
        isFlushScheduled = false;
      }
    };
  };

  const getSnapshot = (): readonly T[] => {
    // Contract: Must return identical reference unless flushed!
    return cachedSnapshot;
  };

  const getStats = () => ({
    totalIngested,
    bufferSize: ringBuffer.length,
    capacity,
    flushCount,
  });

  return {
    push,
    clear,
    subscribe,
    getSnapshot,
    getStats,
  };
}

/**
 * Custom React hook binding RingBufferStore to component render cycle.
 */
export function useRingBuffer<T>(store: RingBufferStore<T>): readonly T[] {
  return useSyncExternalStore(store.subscribe, store.getSnapshot);
}
