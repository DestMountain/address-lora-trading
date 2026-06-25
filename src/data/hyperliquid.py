"""
Hyperliquid API client for fetching on-chain trade data.

Hyperliquid API reference: https://api.hyperliquid.xyz/info
"""

import time
import json
import requests
from typing import Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "https://api.hyperliquid.xyz/info"


@dataclass
class Fill:
    """A single trade fill from Hyperliquid."""
    coin: str
    px: float
    sz: float
    side: str  # 'A' (ask/sell) or 'B' (bid/buy)
    time: int  # ms timestamp
    start_position: float
    dir: str
    closed_pnl: float
    hash: str
    oid: int
    fee: float
    tid: int
    fee_token: str


def _post(endpoint: str, payload: dict) -> Any:
    """Generic POST request to Hyperliquid API."""
    resp = requests.post(
        f"{BASE_URL}/{endpoint}" if endpoint else BASE_URL,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_all_mids() -> dict[str, float]:
    """Get all mid prices."""
    data = _post("", {"type": "allMids"})
    return {k: float(v) for k, v in data.items()}


def get_user_fills(address: str) -> list[Fill]:
    """Get all fills for a user address."""
    data = _post("", {"type": "userFills", "user": address})
    fills = []
    for item in data:
        try:
            fills.append(Fill(
                coin=item.get("coin", ""),
                px=float(item.get("px", 0)),
                sz=float(item.get("sz", 0)),
                side=item.get("side", ""),
                time=int(item.get("time", 0)),
                start_position=float(item.get("startPosition", 0)),
                dir=item.get("dir", ""),
                closed_pnl=float(item.get("closedPnl", 0)),
                hash=item.get("hash", ""),
                oid=int(item.get("oid", 0)),
                fee=float(item.get("fee", 0)),
                tid=int(item.get("tid", 0)),
                fee_token=item.get("feeToken", ""),
            ))
        except (ValueError, KeyError, TypeError) as e:
            continue  # skip malformed entries
    return fills


def get_top_traders_by_volume(coin: str = "ETH", top_n: int = 100) -> list[str]:
    """Get top trader addresses for a given coin.

    Hyperliquid doesn't have a direct 'top traders' endpoint.
    We use an approximation via historical trade data or
    the `clearinghouseState` for open interest.

    For a real implementation, we could:
    1. Query recent trades and extract unique addresses
    2. Or use Hyperliquid's subgraph/indexer
    3. Or use a third-party data provider

    For now, we'll use a curated set of active trader addresses
    from the Hyperliquid ecosystem.
    """
    raise NotImplementedError(
        "Hyperliquid doesn't expose a direct top-trader endpoint. "
        "See fetch_top_addresses() in scripts/fetch_data.py for the workaround."
    )


def fetch_address_fills_batch(
    addresses: list[str],
    max_workers: int = 5,
    rate_limit_delay: float = 0.2,
) -> dict[str, list[Fill]]:
    """Fetch fills for multiple addresses concurrently.

    Args:
        addresses: List of wallet addresses.
        max_workers: Max concurrent requests.
        rate_limit_delay: Delay between requests per worker.

    Returns:
        dict mapping address -> list of Fill objects.
    """
    results: dict[str, list[Fill]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for addr in addresses:
            future = executor.submit(get_user_fills, addr)
            future_map[future] = addr
            time.sleep(rate_limit_delay)

        for future in as_completed(future_map):
            addr = future_map[future]
            try:
                fills = future.result()
                results[addr] = fills
            except Exception as e:
                print(f"  [WARN] Failed to fetch {addr}: {e}")
                results[addr] = []

    return results


def get_spot_asset_ctxs() -> list[dict]:
    """Get spot asset contexts (includes volume information)."""
    data = _post("", {"type": "spotMetaAndAssetCtxs"})
    return data


def get_perp_meta() -> list[dict]:
    """Get perpetual metadata (available markets)."""
    data = _post("", {"type": "perpMeta"})
    return data


def get_open_interest(coin: str) -> float:
    """Get open interest for a coin."""
    data = _post("", {"type": "openInterest", "coin": coin})
    return float(data.get("openInterest", 0))


# ─── Known active addresses (seed set for initial exploration) ───

# These are publicly known active Hyperliquid trader addresses.
# In production, we'd dynamically discover top addresses via
# the exchange's order book or subgraph data.

def discover_active_addresses(
    target_coins: list[str] | None = None,
    max_addresses: int = 100,
) -> list[str]:
    """Discover active trader addresses from recent trades.

    Strategy: query recentTrades for major coins, extract unique users.

    Args:
        target_coins: List of coin symbols to query. Defaults to major pairs.
        max_addresses: Maximum number of unique addresses to return.

    Returns:
        List of unique wallet addresses sorted by activity frequency.
    """
    if target_coins is None:
        target_coins = ["ETH", "BTC", "SOL", "HYPE", "ARB", "OP", "DOGE", "PURR", "AAVE", "LINK"]

    from collections import Counter
    address_counter: Counter = Counter()

    print(f"[*] Discovering active addresses from {len(target_coins)} coins...")
    for coin in target_coins:
        try:
            data = _post("", {
                "type": "recentTrades",
                "coin": coin,
            })
            for trade in data:
                for user in trade.get("users", []):
                    if isinstance(user, str) and user.startswith("0x"):
                        address_counter[user] += 1
        except Exception as e:
            print(f"  [WARN] Failed to fetch trades for {coin}: {e}")

    # Remove zero address
    zero_addr = "0x0000000000000000000000000000000000000000"
    address_counter.pop(zero_addr, None)

    top = address_counter.most_common(max_addresses)
    print(f"    Found {len(address_counter)} unique addresses, taking top {min(max_addresses, len(top))}")

    addresses = [addr for addr, _ in top]
    return addresses
