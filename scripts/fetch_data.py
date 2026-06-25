#!/usr/bin/env python3
"""
Fetch Hyperliquid trade data for active addresses.

Usage:
    python scripts/fetch_data.py [--top-n 100] [--days 90] [--output data/raw]
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hyperliquid import (
    discover_active_addresses,
    fetch_address_fills_batch,
    get_all_mids,
)


def fetch_and_save(
    addresses: list[str],
    output_dir: str | Path,
    rate_limit: float = 0.25,
    max_workers: int = 3,
):
    """Fetch fills for addresses and save to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Fetching {len(addresses)} addresses (rate_limit={rate_limit}s, workers={max_workers})...")

    results = fetch_address_fills_batch(
        addresses=addresses,
        max_workers=max_workers,
        rate_limit_delay=rate_limit,
    )

    # Save individual address data
    meta = {
        "fetch_time": int(time.time()),
        "total_addresses": len(addresses),
        "addresses_with_data": 0,
        "total_fills": 0,
        "address_files": [],
    }

    for addr, fills in results.items():
        if not fills:
            continue

        meta["addresses_with_data"] += 1
        meta["total_fills"] += len(fills)

        # Save as JSON
        short_addr = addr[:12]
        filename = output_dir / f"{short_addr}.json"
        with open(filename, "w") as f:
            json.dump(
                [fill.__dict__ for fill in fills],
                f,
                indent=2,
                default=str,
            )
        meta["address_files"].append(str(filename))

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[*] Done!")
    print(f"    Addresses with data: {meta['addresses_with_data']}/{meta['total_addresses']}")
    print(f"    Total fills fetched: {meta['total_fills']}")
    print(f"    Data saved to: {output_dir.resolve()}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid trade data")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top addresses to fetch")
    parser.add_argument("--days", type=int, default=90, help="Lookback period in days (not directly supported by HL API)")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--rate-limit", type=float, default=0.25, help="Delay between API calls (seconds)")
    parser.add_argument("--workers", type=int, default=3, help="Max concurrent workers")
    args = parser.parse_args()

    # Snapshot current market
    print("[*] Snapshotting current market...")
    mids = get_all_mids()
    print(f"    Found {len(mids)} trading pairs")

    # Discover addresses
    addresses = discover_active_addresses(max_addresses=args.top_n)
    print(f"[*] Will fetch {len(addresses)} addresses")

    # Fetch
    fetch_and_save(
        addresses=addresses,
        output_dir=args.output,
        rate_limit=args.rate_limit,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
