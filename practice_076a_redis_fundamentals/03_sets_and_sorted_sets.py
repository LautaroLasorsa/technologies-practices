# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Sets & Sorted Sets -- Set Operations and Leaderboard Pattern.

Demonstrates:
  - SADD, SMEMBERS, SISMEMBER, SCARD, SRANDMEMBER
  - Set operations: SINTER, SUNION, SDIFF
  - ZADD, ZSCORE, ZRANK, ZRANGE, ZREVRANGE, ZINCRBY
  - Leaderboard pattern with sorted sets

Run after starting Redis:
    uv run 03_sets_and_sorted_sets.py
"""

from __future__ import annotations

import redis


def get_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def cleanup(r: redis.Redis, prefix: str) -> None:
    for key in r.scan_iter(f"{prefix}*"):
        r.delete(key)


# -- TODO(human): Implement these functions --------------------------------


def set_operations(r: redis.Redis) -> None:
    """Exercise 1: Redis Set operations.

    TODO(human): Implement this function.

    Redis Sets store unique, unordered string members. They're ideal for:
      - Tracking unique visitors, IP addresses, or tags
      - Set math: find common friends, shared interests, permission overlaps
      - Random sampling (SRANDMEMBER) for features like "random suggestion"

    Internally, small all-integer sets use an intset (sorted array of ints,
    very memory efficient). Larger or mixed-type sets use a hashtable.

    Steps:

    1. Add members to a set "languages:alice":
           r.sadd("languages:alice", "Python", "Rust", "C++", "JavaScript")
       SADD adds one or more members. Returns the number of NEW members added
       (ignores duplicates). Print the return value.

    2. Add members to another set "languages:bob":
           r.sadd("languages:bob", "Python", "Go", "JavaScript", "TypeScript")

    3. Check set membership:
           is_member = r.sismember("languages:alice", "Rust")
       SISMEMBER returns True if the member exists, False otherwise. O(1).
       Check a few members and print results.

    4. Get all members of a set:
           members = r.smembers("languages:alice")
       SMEMBERS returns a Python set. Print it.
       Note: the order is NOT guaranteed (sets are unordered).

    5. Get set cardinality (size):
           count = r.scard("languages:alice")
       SCARD is O(1). Print it.

    6. SINTER -- intersection (languages BOTH Alice and Bob know):
           common = r.sinter("languages:alice", "languages:bob")
       Print the result. This is useful for "mutual friends" or "shared tags".

    7. SUNION -- union (ALL languages between Alice and Bob):
           all_langs = r.sunion("languages:alice", "languages:bob")
       Print the result.

    8. SDIFF -- difference (languages Alice knows but Bob doesn't):
           alice_only = r.sdiff("languages:alice", "languages:bob")
       Print the result. Note: SDIFF is directional (A - B != B - A).
       Also compute Bob's exclusive languages:
           bob_only = r.sdiff("languages:bob", "languages:alice")
       Print it to compare.

    9. SRANDMEMBER -- get random members without removing:
           random_one = r.srandmember("languages:alice")
           random_three = r.srandmember("languages:alice", 2)
       SRANDMEMBER with count returns a list of N random unique members.
       With negative count, it may return duplicates. Print both results.

    10. SREM -- remove a member:
            removed = r.srem("languages:alice", "JavaScript")
        Returns the number of members actually removed. Print it and
        then SMEMBERS again to confirm.

    Docs:
      - SADD: https://redis.io/docs/latest/commands/sadd/
      - SINTER: https://redis.io/docs/latest/commands/sinter/
      - SDIFF: https://redis.io/docs/latest/commands/sdiff/
    """
    raise NotImplementedError("TODO(human): implement set_operations")


def leaderboard(r: redis.Redis) -> None:
    """Exercise 2: Sorted set leaderboard pattern.

    TODO(human): Implement this function.

    Sorted Sets (ZSETs) are one of Redis's most powerful and unique
    structures. Each member has a floating-point score, and the set is
    always sorted by score. Internally, this uses a skip list (O(log N)
    insert/rank) paired with a hashtable (O(1) score lookup by member).

    The leaderboard is the canonical sorted set use case: gaming scores,
    trending content, rate limiting counters, priority queues. Production
    systems at Twitter, Riot Games, and Discord use Redis sorted sets for
    real-time rankings.

    Steps:

    1. Add players to a leaderboard with initial scores:
           r.zadd("leaderboard", {
               "alice": 1500,
               "bob": 1200,
               "charlie": 1800,
               "diana": 1350,
               "eve": 1650,
           })
       ZADD adds members with scores. If a member already exists, the score
       is updated. Returns the number of NEW members added (not updates).
       Print the return value.

    2. Get a player's score:
           score = r.zscore("leaderboard", "alice")
       ZSCORE returns a float, or None if the member doesn't exist. Print it.

    3. Get the top 3 players (highest scores):
           top3 = r.zrevrange("leaderboard", 0, 2, withscores=True)
       ZREVRANGE returns members in descending score order.
       With withscores=True, it returns a list of (member, score) tuples.
       Print the top 3 with their scores.

    4. Get a player's rank (0-indexed, ascending):
           rank = r.zrevrank("leaderboard", "alice")
       ZREVRANK returns the rank in descending order (0 = highest score).
       ZRANK returns the rank in ascending order (0 = lowest score).
       Print Alice's rank (0-indexed) and display as "place #N" (1-indexed).

    5. Increment a player's score (simulate a game win):
           new_score = r.zincrby("leaderboard", 200, "bob")
       ZINCRBY atomically adds to the member's score. Returns the new score.
       Print it. Then get the updated top 3 to see if Bob's rank changed.

    6. Get players within a score range:
           mid_range = r.zrangebyscore("leaderboard", 1300, 1600, withscores=True)
       ZRANGEBYSCORE returns members whose scores fall in [min, max].
       Print the results. This is useful for tier-based queries
       ("show all Gold-tier players").

    7. Count members in a score range:
           count = r.zcount("leaderboard", 1300, 1600)
       ZCOUNT is O(log N). Print it.

    8. Remove a player:
           removed = r.zrem("leaderboard", "eve")
       ZREM returns the number of members removed. Print the final
       leaderboard with ZREVRANGE to confirm.

    Docs:
      - ZADD: https://redis.io/docs/latest/commands/zadd/
      - ZREVRANGE: https://redis.io/docs/latest/commands/zrevrange/
      - ZRANGEBYSCORE: https://redis.io/docs/latest/commands/zrangebyscore/
      - ZINCRBY: https://redis.io/docs/latest/commands/zincrby/
    """
    raise NotImplementedError("TODO(human): implement leaderboard")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    cleanup(r, "languages:")
    cleanup(r, "leaderboard")

    section("Exercise 1: Set Operations")
    set_operations(r)

    section("Exercise 2: Sorted Set Leaderboard")
    leaderboard(r)

    print("\n--- All set/sorted set exercises completed ---")


if __name__ == "__main__":
    main()
