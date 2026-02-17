"""Shared configuration for Kafka practice 003e.

Defines broker connection settings, topic names, and consumer group IDs
used across all scripts in this practice. All three brokers are listed
in BOOTSTRAP_SERVERS so the client can fail over if one is unreachable.
"""

# -- Broker connection (multi-broker cluster) ---------------------------------

BOOTSTRAP_SERVERS = "localhost:9092,localhost:9093,localhost:9094"

# -- Topic names --------------------------------------------------------------

REPLICATION_TOPIC = "replication-demo"
REPLICATION_PARTITIONS = 3

COMPRESSION_TOPIC = "compression-bench"
COMPRESSION_PARTITIONS = 3

TRANSACTION_TOPIC = "transaction-demo"
TRANSACTION_PARTITIONS = 3

CONSUMER_TUNING_TOPIC = "consumer-tuning"
CONSUMER_TUNING_PARTITIONS = 6

COMPACTION_TOPIC = "compaction-demo"
COMPACTION_PARTITIONS = 1

# -- Consumer group IDs -------------------------------------------------------

REPLICATION_GROUP = "replication-explorer-group"
TRANSACTION_GROUP = "transaction-consumer-group"
CONSUMER_TUNING_GROUP = "consumer-tuning-group"
LAG_MONITOR_GROUP = "lag-monitor-group"
COMPACTION_GROUP = "compaction-consumer-group"
