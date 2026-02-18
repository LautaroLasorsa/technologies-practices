//! Exercise 1: crossbeam-channel & select!
//!
//! crossbeam-channel provides multi-producer multi-consumer (MPMC) channels,
//! a significant upgrade over std::sync::mpsc which only supports a single
//! consumer. It also supports bounded channels (with backpressure), zero-capacity
//! rendezvous channels, and the `select!` macro for multiplexing.

use crossbeam_channel::{self as channel, select, Receiver, Sender};
use std::time::Duration;

/// Fan-out / fan-in pattern using a bounded channel.
///
/// Architecture:
///   N producers --> [bounded channel (capacity)] --> M consumers --> [results channel] --> collector
///
/// The bounded channel provides **backpressure**: if the channel is full, producers
/// block on send() until a consumer pops an item. This prevents fast producers from
/// overwhelming slow consumers and consuming unbounded memory.
///
/// crossbeam::scope() creates scoped threads that can borrow from the enclosing stack
/// frame — no need for Arc or 'static lifetimes. The scope blocks until all spawned
/// threads complete, making cleanup automatic.
pub fn fan_out_fan_in() {
    let num_producers = 4;
    let num_consumers = 2;
    let items_per_producer = 100;
    let channel_capacity = 10; // Small capacity to demonstrate backpressure

    // TODO(human): Implement the fan-out / fan-in pattern.
    //
    // This is a fundamental concurrent architecture pattern used everywhere from
    // web servers (request queue → worker pool) to data pipelines (partition → process → merge).
    //
    // Steps:
    //
    // 1. Create a bounded work channel: `channel::bounded::<u64>(channel_capacity)`
    //    This returns (sender, receiver). The bounded capacity means send() will BLOCK
    //    when the channel has `channel_capacity` pending items — this is backpressure.
    //    Compare with `channel::unbounded()` which never blocks but can consume unlimited memory.
    //
    // 2. Create an unbounded results channel: `channel::unbounded::<u64>()`
    //    Results are collected here. Unbounded is fine because consumers produce results
    //    at most as fast as they consume work items.
    //
    // 3. Use `crossbeam::scope(|s| { ... })` to spawn scoped threads. Inside the scope:
    //
    //    a) Spawn `num_producers` producer threads. Each producer:
    //       - Sends `items_per_producer` items via `work_sender.send(value).unwrap()`
    //       - Items should be unique (e.g., producer_id * items_per_producer + i)
    //       - After sending all items, the thread exits (dropping its clone of the sender)
    //       NOTE: Clone the sender for each producer thread. When ALL senders are dropped,
    //       the channel closes and receivers' recv() returns Err.
    //
    //    b) Drop the original work_sender after spawning producers (so the channel
    //       closes when all producers finish, not when main finishes):
    //       `drop(work_sender);`
    //
    //    c) Spawn `num_consumers` consumer threads. Each consumer:
    //       - Loops: `while let Ok(item) = work_receiver.recv() { ... }`
    //       - Processes each item (e.g., item * 2) and sends result to results channel
    //       - The loop exits automatically when all senders are dropped (channel closed)
    //
    //    d) Drop the original results sender after spawning consumers.
    //
    //    e) Collect all results from the results receiver into a Vec.
    //       Loop: `while let Ok(result) = results_receiver.recv() { ... }`
    //
    // 4. After the scope exits (all threads joined), verify:
    //    - results.len() == num_producers * items_per_producer
    //    - No items were lost or duplicated
    //
    // Key insight: The bounded channel capacity (10) is much smaller than total items
    // (4 * 100 = 400). Producers MUST wait when the channel is full, and consumers
    // MUST process items to unblock producers. This is cooperative flow control —
    // the same principle behind TCP flow control and Kafka consumer groups.

    todo!("Exercise 1a: Implement fan-out/fan-in with bounded crossbeam channel")
}

/// Multiplexing channels with the `select!` macro.
///
/// The select! macro waits on multiple channel operations simultaneously and
/// executes whichever becomes ready first. This is analogous to:
/// - Go's `select { case msg := <-ch1: ... case msg := <-ch2: ... }`
/// - Unix `select()/poll()/epoll()` for file descriptors
/// - async Rust's `tokio::select!` for futures
///
/// Unlike std::sync::mpsc which has no select mechanism, crossbeam-channel's
/// select! supports: recv from multiple channels, send to multiple channels,
/// timeouts via `default(duration)`, and non-blocking attempts via `default()`.
pub fn select_timeout() {
    // TODO(human): Implement a select! loop that multiplexes two channels with a timeout.
    //
    // This pattern is essential for building event loops, multiplexers, and any system
    // that must respond to multiple event sources without dedicating a thread per source.
    //
    // Steps:
    //
    // 1. Create two unbounded channels: `fast_channel` and `slow_channel`.
    //
    // 2. Spawn a thread for the "fast" producer:
    //    - Sends 5 messages with 50ms delay between each
    //    - Messages like "fast-0", "fast-1", etc.
    //
    // 3. Spawn a thread for the "slow" producer:
    //    - Sends 3 messages with 200ms delay between each
    //    - Messages like "slow-0", "slow-1", etc.
    //
    // 4. In the main thread, run a select! loop that processes messages from both
    //    channels until both are exhausted:
    //
    //    ```
    //    loop {
    //        select! {
    //            recv(fast_receiver) -> msg => {
    //                match msg {
    //                    Ok(m) => println!("  [fast] {}", m),
    //                    Err(_) => { /* channel closed, mark fast as done */ }
    //                }
    //            },
    //            recv(slow_receiver) -> msg => {
    //                match msg {
    //                    Ok(m) => println!("  [slow] {}", m),
    //                    Err(_) => { /* channel closed, mark slow as done */ }
    //                }
    //            },
    //            default(Duration::from_millis(300)) => {
    //                println!("  [timeout] No message in 300ms");
    //            }
    //        }
    //        // Break when both channels are closed
    //    }
    //    ```
    //
    //    The `default(Duration)` arm fires if no channel has a message within the timeout.
    //    This is useful for implementing heartbeat checks, idle detection, or graceful shutdown.
    //
    // 5. Print a summary of how many messages were received from each channel.
    //
    // Key insight: select! chooses ONE ready operation per iteration. If both channels
    // have messages, one is chosen at random (fair scheduling). This prevents starvation
    // where a fast channel could monopolize the receiver.
    //
    // Note: When a channel is closed (all senders dropped), recv() returns Err.
    // You can handle this by replacing the receiver with `channel::never()` —
    // a special receiver that never produces messages, effectively disabling that
    // arm of the select without restructuring the loop.

    todo!("Exercise 1b: Implement select! multiplexing with timeout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fan_out_fan_in() {
        // Should not panic — verifies no items lost
        fan_out_fan_in();
    }

    #[test]
    fn test_select_timeout() {
        // Should not panic — verifies clean shutdown
        select_timeout();
    }
}
