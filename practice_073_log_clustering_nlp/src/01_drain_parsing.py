#!/usr/bin/env python3
"""Exercise 1-2: Parse raw logs with Drain3 template extraction.

Pipeline:
  1. Load raw logs from data/logs.txt
  2. Configure a Drain3 TemplateMiner with masking for distributed system logs
  3. Feed each log line through the miner to extract templates
  4. Save parsed results to data/parsed_logs.csv

After running, you should see 40-80 unique templates from the 10,000+ log lines.
If you see hundreds, masking is too loose; if fewer than 20, sim_th is too low.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# TODO(human): Exercise 1 -- Configure Drain3 TemplateMiner
# ---------------------------------------------------------------------------
def configure_drain():
    """Create and return a configured drain3.TemplateMiner.

    TODO(human): Import and configure drain3 for distributed system log parsing.

    Steps:
      1. Import TemplateMiner and TemplateMinerConfig from drain3.
      2. Create a TemplateMinerConfig instance.
      3. Set drain parameters:
         - config.drain_sim_th = 0.4   (similarity threshold)
           Controls how similar a log line must be to an existing template
           to join that cluster. Range [0, 1]. Lower = more aggressive merging.
           Try 0.4 first, then experiment: 0.3 merges too aggressively (unrelated
           logs share templates), 0.6 over-splits (minor variations create new templates).
         - config.drain_depth = 4      (parse tree depth)
         - config.drain_max_children = 100
      4. Configure masking instructions to replace variable parts BEFORE
         template mining. Without masking, every unique IP address, UUID,
         or number creates a separate template -- defeating the purpose.

         Create a list of RegexMaskingInstruction objects for:
           - IP addresses:  r"(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})"  -> "IP"
           - UUIDs:         r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"  -> "UUID"
           - Email addrs:   r"([\\w.+-]+@[\\w-]+\\.[\\w.]+)"  -> "EMAIL"
           - Phone numbers: r"(\\+\\d{10,15})"  -> "PHONE"
           - File paths:    r"(/[\\w./\\-]+)"  -> "PATH"
           - Hex tokens:    r"\\b([0-9a-f]{8,})\\b"  -> "HEX"
           - Numbers:       r"((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)"  -> "NUM"

         Use: from drain3.masking import MaskingInstruction (or RegexMaskingInstruction)
         Each MaskingInstruction takes (regex_pattern: str, mask_with: str).
         Assign the list to config.masking_instructions.

         ORDER MATTERS: More specific patterns (IP, UUID) should come before
         generic ones (NUM). Otherwise NUM matches the octets in an IP first.
      5. Create TemplateMiner(config=config) and return it.

    Returns:
        A configured drain3.TemplateMiner ready to process log lines.

    Hint: The mask_prefix and mask_suffix default to "<" and ">" respectively,
    so masked IPs appear as <IP> in templates. This is the standard convention.
    """
    raise NotImplementedError("TODO(human): Configure Drain3 TemplateMiner")


# ---------------------------------------------------------------------------
# TODO(human): Exercise 2 -- Parse logs through the TemplateMiner
# ---------------------------------------------------------------------------
def parse_logs(
    miner,
    log_lines: list[str],
) -> pd.DataFrame:
    """Feed each log line through the Drain3 miner and collect results.

    TODO(human): Process each log line with the miner's add_log_message() method.

    Steps:
      1. Iterate over log_lines.
      2. For each line, call: result = miner.add_log_message(line)
         This returns a dict with keys:
           - "cluster_id"      : int, the template cluster this line was assigned to
           - "template_mined"  : str, the current template for that cluster
           - "change_type"     : str, one of "cluster_created", "cluster_template_changed", "none"
           - "cluster_size"    : int, how many lines are in this cluster so far
           - "cluster_count"   : int, total number of clusters seen so far
      3. Collect a list of dicts, one per log line, with keys:
         ["line_idx", "cluster_id", "template", "raw_log"]
      4. Print progress every 1000 lines:
         print(f"  Processed {i+1}/{len(log_lines)} lines, clusters so far: {result['cluster_count']}")
      5. Return a pandas DataFrame from the collected list of dicts.

    The streaming nature of Drain means templates evolve: early lines may create
    new clusters ("cluster_created"), while later lines with similar patterns
    merge into existing clusters ("none" = matched existing template).

    Args:
        miner: A configured drain3.TemplateMiner (from configure_drain).
        log_lines: List of raw log strings (one per line from logs.txt).

    Returns:
        pd.DataFrame with columns: [line_idx, cluster_id, template, raw_log]
    """
    raise NotImplementedError("TODO(human): Parse logs through Drain3 miner")


# ---------------------------------------------------------------------------
# Scaffolded: main pipeline
# ---------------------------------------------------------------------------
def print_template_summary(parsed_df: pd.DataFrame) -> None:
    """Print the top-20 most frequent templates."""
    template_counts = (
        parsed_df.groupby(["cluster_id", "template"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\n{'='*80}")
    print(f"TEMPLATE SUMMARY: {len(template_counts)} unique templates extracted")
    print(f"{'='*80}")

    print(f"\nTop 20 templates by frequency:")
    print(f"{'-'*80}")
    for i, row in template_counts.head(20).iterrows():
        print(f"  [{row['cluster_id']:>4}] (n={row['count']:>5}) {row['template'][:100]}")

    # Distribution stats
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION STATS")
    print(f"{'='*80}")
    print(f"  Total log lines:     {len(parsed_df)}")
    print(f"  Unique templates:    {len(template_counts)}")
    print(f"  Largest cluster:     {template_counts['count'].max()} lines")
    print(f"  Smallest cluster:    {template_counts['count'].min()} lines")
    print(f"  Median cluster size: {template_counts['count'].median():.0f} lines")

    # Show a few templates with low frequency (potential anomalies)
    rare = template_counts[template_counts["count"] <= 5]
    if len(rare) > 0:
        print(f"\n  Rare templates (count <= 5): {len(rare)}")
        for _, row in rare.head(5).iterrows():
            print(f"    [{row['cluster_id']:>4}] (n={row['count']}) {row['template'][:90]}")


def main() -> None:
    """Run the Drain3 parsing pipeline."""
    logs_path = DATA_DIR / "logs.txt"
    if not logs_path.exists():
        print("ERROR: data/logs.txt not found. Run 00_generate_logs.py first.")
        return

    # Load raw logs
    print(f"Loading logs from {logs_path}...")
    with open(logs_path, "r", encoding="utf-8") as f:
        log_lines = [line.rstrip("\n") for line in f if line.strip()]
    print(f"  Loaded {len(log_lines)} log lines")

    # Configure Drain3
    print("\nConfiguring Drain3 TemplateMiner...")
    miner = configure_drain()
    print("  Miner configured.")

    # Parse logs
    print(f"\nParsing {len(log_lines)} log lines through Drain3...")
    parsed_df = parse_logs(miner, log_lines)

    # Summary
    print_template_summary(parsed_df)

    # Save results
    output_path = DATA_DIR / "parsed_logs.csv"
    parsed_df.to_csv(output_path, index=False)
    print(f"\nSaved parsed results to {output_path}")

    # Also save the unique templates for the next step
    templates_path = DATA_DIR / "unique_templates.csv"
    unique = (
        parsed_df.groupby(["cluster_id", "template"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    unique.to_csv(templates_path, index=False)
    print(f"Saved {len(unique)} unique templates to {templates_path}")


if __name__ == "__main__":
    main()
