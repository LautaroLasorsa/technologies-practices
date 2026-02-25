# /// script
# requires-python = ">=3.12"
# dependencies = ["redis>=5.0"]
# ///
"""Redis Persistence -- Inspecting and Exploring RDB/AOF Configuration.

Demonstrates:
  - CONFIG GET to inspect persistence settings
  - BGSAVE to trigger a manual RDB snapshot
  - LASTSAVE to check last snapshot timestamp
  - INFO persistence for RDB/AOF status
  - Understanding the persistence configuration in docker-compose.yml

Run after starting Redis:
    uv run 09_persistence.py
"""

from __future__ import annotations

import datetime
import time

import redis


def get_client() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# -- TODO(human): Implement these functions --------------------------------


def explore_persistence(r: redis.Redis) -> None:
    """Exercise: Inspect and explore Redis persistence settings.

    TODO(human): Implement this function.

    Understanding persistence configuration is critical for production
    Redis deployments. The docker-compose.yml in this practice starts
    Redis with both RDB and AOF enabled. This exercise teaches you how
    to inspect, verify, and reason about those settings.

    Our docker-compose.yml starts Redis with:
      --appendonly yes             (AOF enabled)
      --appendfsync everysec      (fsync every second)
      --save 3600 1               (RDB: snapshot every 3600s if >= 1 change)
      --save 300 100              (RDB: snapshot every 300s if >= 100 changes)
      --save 60 10000             (RDB: snapshot every 60s if >= 10000 changes)

    Steps:

    1. Check if AOF is enabled:
           aof_enabled = r.config_get("appendonly")
       CONFIG GET returns a dict. Print the result. The value should be "yes"
       because we passed --appendonly yes in docker-compose.yml.

    2. Check the AOF fsync policy:
           fsync_policy = r.config_get("appendfsync")
       Print it. "everysec" means Redis calls fsync() once per second --
       at most 1 second of data can be lost on crash.

    3. Check the RDB save schedule:
           save_config = r.config_get("save")
       CONFIG GET "save" returns the save schedule as a single string like
       "3600 1 300 100 60 10000" (pairs of seconds and changes).
       Print it and explain what each pair means.

    4. Check the RDB filename:
           dbfilename = r.config_get("dbfilename")
       Print it. Default is "dump.rdb".

    5. Check the data directory:
           dir_config = r.config_get("dir")
       This is where Redis stores .rdb and .aof files. Print it.

    6. Get the last successful RDB save timestamp:
           last_save = r.lastsave()
       LASTSAVE returns a datetime object (in redis-py with decode_responses).
       Print it.

    7. Trigger a manual background save:
           r.bgsave()
       BGSAVE forks a child process to write the RDB file. It returns
       immediately (the save happens in the background). Print "BGSAVE
       initiated".

       Wait a moment and check if it completed:
           time.sleep(1)
           new_last_save = r.lastsave()
       Compare with the previous LASTSAVE timestamp.

    8. Get detailed persistence stats from INFO:
           info = r.info("persistence")
       INFO returns a dict with many fields. Print the interesting ones:
         - info["rdb_last_save_time"]     -- Unix timestamp of last RDB save
         - info["rdb_last_bgsave_status"] -- "ok" if last save succeeded
         - info["rdb_changes_since_last_save"] -- number of changes since last RDB
         - info["aof_enabled"]            -- 1 if AOF is on
         - info["aof_rewrite_in_progress"] -- 1 if AOF rewrite is happening
         - info["aof_last_bgrewrite_status"] -- "ok" if last AOF rewrite succeeded

       Loop through these keys and print each with its value.

    9. Get memory usage info:
           info_memory = r.info("memory")
       Print:
         - info_memory["used_memory_human"]   -- e.g., "1.5M"
         - info_memory["maxmemory_human"]     -- e.g., "0" (no limit set)
         - info_memory["maxmemory_policy"]    -- e.g., "noeviction"
       Explain: when maxmemory is 0, Redis has no memory limit (uses all
       available RAM). In production, you MUST set maxmemory and choose
       an appropriate eviction policy.

    10. Write some data and observe rdb_changes_since_last_save increase:
            for i in range(10):
                r.set(f"persist_test:{i}", f"value-{i}")
            info2 = r.info("persistence")
            print(f"  Changes since last save: {info2['rdb_changes_since_last_save']}")
        Then trigger another BGSAVE and check:
            r.bgsave()
            time.sleep(1)
            info3 = r.info("persistence")
            print(f"  Changes after BGSAVE: {info3['rdb_changes_since_last_save']}")
        The count should have reset to 0 (or close to it).

        Clean up test keys:
            for i in range(10):
                r.delete(f"persist_test:{i}")

    Docs:
      - CONFIG GET: https://redis.io/docs/latest/commands/config-get/
      - BGSAVE: https://redis.io/docs/latest/commands/bgsave/
      - LASTSAVE: https://redis.io/docs/latest/commands/lastsave/
      - INFO: https://redis.io/docs/latest/commands/info/
      - Persistence docs: https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/
    """
    raise NotImplementedError("TODO(human): implement explore_persistence")


# -- Orchestration (boilerplate) -------------------------------------------


def main() -> None:
    r = get_client()
    r.ping()
    print("Connected to Redis!")

    section("Exercise: Redis Persistence Exploration")
    explore_persistence(r)

    print("\n--- All persistence exercises completed ---")


if __name__ == "__main__":
    main()
