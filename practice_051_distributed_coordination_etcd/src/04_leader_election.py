"""Exercise 4: Leader Election.

This exercise implements leader election using etcd: multiple candidates
compete to become leader by acquiring a key via CAS transaction. The winner
maintains leadership via lease keepalive. When the leader crashes or
resigns, watchers detect the key deletion and trigger a new election.

Leader election is a fundamental distributed systems primitive used in:
- Kubernetes controller manager (only one active at a time)
- Kafka controller (one broker is the controller)
- Distributed schedulers (one scheduler assigns work)
- Database primary election (one node accepts writes)

This exercise combines all previous concepts: leases (auto-expire on crash),
transactions (CAS for election), and watches (detect leadership changes).
"""

import threading
import time

import etcd3


ETCD_HOST = "localhost"
ETCD_PORT = 2379

ELECTION_PREFIX = "/election/"


def create_client() -> etcd3.Etcd3Client:
    """Create an etcd client connected to the first cluster node."""
    return etcd3.client(host=ETCD_HOST, port=ETCD_PORT)


def cleanup(client: etcd3.Etcd3Client) -> None:
    """Delete all election keys to start fresh."""
    client.delete_prefix(ELECTION_PREFIX)


# TODO(human): Implement the LeaderElection class
#
# A leader election protocol built on etcd leases, transactions, and watches.
# Multiple candidates can run campaign() concurrently — exactly one will
# become leader at any time. If the leader crashes (lease expires) or
# resigns, a new election occurs automatically.
#
# Class state:
#   - client: etcd3 client (each candidate should use its own client)
#   - election_name: name of the election (used to build the key)
#   - candidate_id: unique ID for this candidate (e.g., "candidate-1")
#   - ttl_secs: lease TTL for the leadership key
#   - election_key: f"{ELECTION_PREFIX}{election_name}" (the key candidates compete for)
#   - lease: the lease backing our leadership (None until we win)
#   - is_leader: bool flag indicating if we currently hold leadership
#   - stop_event: threading.Event to signal shutdown
#   - keepalive_thread: background thread for lease keepalive
#
# Implement these methods:
#
# __init__(self, client, election_name, candidate_id, ttl_secs=10):
#     Initialize all state. Set is_leader=False, lease=None.
#
# campaign(self, timeout_secs=60.0) -> bool:
#     Attempt to become leader. The algorithm:
#
#     1. Create a lease with the given TTL.
#
#     2. Execute a CAS transaction:
#        Compare:  [version(election_key) == 0]  (no current leader)
#        Success:  [put(election_key, candidate_id, lease)]  (we become leader)
#        Failure:  [get(election_key)]  (someone else is leader)
#
#     3. If we won: set is_leader=True, start a keepalive thread that
#        calls lease.refresh() every ttl_secs/3 seconds. Print a victory
#        message with timestamp. Return True.
#
#     4. If we lost: extract the current leader's ID from the failure
#        response. Print who the current leader is. Then watch the
#        election_key — when it's deleted (leader crashed/resigned),
#        retry from step 1.
#
#     5. Repeat until we become leader, timeout, or stop_event is set.
#
#     The watch-and-retry pattern avoids busy-spinning. When the current
#     leader's key disappears, all waiting candidates race to create it.
#     The CAS transaction ensures exactly one wins.
#
# resign(self) -> None:
#     Voluntarily give up leadership:
#     1. Set is_leader = False
#     2. Stop the keepalive thread (stop_event.set())
#     3. Delete the election key
#     4. Revoke the lease
#     Print a resignation message with timestamp.
#     Handle errors gracefully (key/lease may already be gone).
#
# observe(self) -> str | None:
#     Check who the current leader is (without trying to become leader).
#     Read the election_key — if it exists, return the leader's candidate_id.
#     If no leader, return None.
#     This is useful for non-candidate observers (e.g., a web UI showing
#     which node is the current primary).
#
# Use:
#   client.transaction(compare=[...], success=[...], failure=[...])
#   client.transactions.version(key) == 0
#   client.transactions.put(key, value, lease)
#   client.transactions.get(key)
#   client.watch(key) -> (events_iterator, cancel)
#   lease.refresh() for keepalive
#   lease.revoke() to destroy lease
#   time.strftime("%H:%M:%S") for timestamps
#
# Class signature:
#   class LeaderElection:
#       def __init__(self, client: etcd3.Etcd3Client, election_name: str,
#                    candidate_id: str, ttl_secs: int = 10) -> None
#       def campaign(self, timeout_secs: float = 60.0) -> bool
#       def resign(self) -> None
#       def observe(self) -> str | None
class LeaderElection:
    def __init__(
        self,
        client: etcd3.Etcd3Client,
        election_name: str,
        candidate_id: str,
        ttl_secs: int = 10,
    ) -> None:
        raise NotImplementedError("TODO(human): Implement LeaderElection.__init__")

    def campaign(self, timeout_secs: float = 60.0) -> bool:
        raise NotImplementedError("TODO(human): Implement LeaderElection.campaign")

    def resign(self) -> None:
        raise NotImplementedError("TODO(human): Implement LeaderElection.resign")

    def observe(self) -> str | None:
        raise NotImplementedError("TODO(human): Implement LeaderElection.observe")


# TODO(human): Implement simulate_election
#
# Run a full election simulation with multiple candidates and leader failover.
# This ties together everything: multiple candidates campaign concurrently,
# exactly one wins, the rest wait. When the leader resigns/crashes, a new
# leader is elected.
#
# Steps:
#   1. Create num_candidates candidates, each with:
#      - Its own etcd client (thread safety — one client per thread)
#      - A LeaderElection instance with a unique candidate_id (f"candidate-{i}")
#      - A thread that runs campaign()
#
#   2. Start all candidate threads simultaneously.
#      Wait a few seconds for one candidate to win.
#
#   3. Identify the winner by checking is_leader on each candidate.
#      Print the winner's ID.
#
#   4. Use observe() from any candidate to confirm the leader externally.
#
#   5. Simulate leader failure: call resign() on the winning candidate.
#      This deletes the election key and revokes the lease.
#
#   6. Wait for a new leader to be elected (the watching candidates should
#      detect the deletion and retry their campaign). Print the new leader.
#
#   7. Clean up: resign all remaining candidates, set stop_events,
#      join all threads.
#
# Print a timeline of events with timestamps so the user can see the
# full lifecycle: election -> leader serves -> leader resigns -> new election.
#
# Use:
#   threading.Thread(target=candidate.campaign) for each candidate
#   time.sleep() for pacing
#   candidate.is_leader to check who won
#   candidate.resign() to trigger failover
#   candidate.observe() to check current leader
#
# Function signature:
#   def simulate_election(
#       num_candidates: int,
#       election_name: str,
#   ) -> None
def simulate_election(
    num_candidates: int,
    election_name: str,
) -> None:
    raise NotImplementedError("TODO(human): Implement simulate_election")


def main() -> None:
    print("=" * 60)
    print("  Exercise 4: Leader Election")
    print("=" * 60)

    client = create_client()
    cleanup(client)
    client.close()

    # --- Run election with 3 candidates ---
    print("\n--- Election with 3 candidates ---\n")
    simulate_election(num_candidates=3, election_name="primary-db")

    # --- Final cleanup ---
    client = create_client()
    cleanup(client)
    client.close()

    print("\n" + "=" * 60)
    print("  Exercise 4 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
