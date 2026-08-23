"""
data/synthetic_data.py

Seeded synthetic retail dataset for exercising ACOS end to end: product
catalog, 12 weeks of per-product sales history, current inventory,
candidate suppliers, and a batch of transactions with a known-injected
fraction of anomalies for measuring fraud-detection recall against
ground truth.

Synthetic and explicitly labeled as such -- not real enterprise data.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

PRODUCT_CATEGORIES = ["electronics", "home_goods", "apparel", "grocery", "sporting_goods"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU"]


def generate_catalog(n_products: int = 20, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    rng = random.Random(seed)
    catalog = {}
    for i in range(n_products):
        pid = f"SKU-{i+1:03d}"
        unit_cost = round(rng.uniform(5, 150), 2)
        margin_target = rng.uniform(1.3, 2.2)
        catalog[pid] = {
            "category": rng.choice(PRODUCT_CATEGORIES),
            "unit_cost": unit_cost,
            "current_price": round(unit_cost * margin_target, 2),
            "elasticity": round(rng.uniform(-2.2, -0.8), 2),
        }
    return catalog


def generate_sales_history(catalog: Dict[str, Dict[str, Any]], weeks: int = 12,
                            seed: int = 7) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    history = {}
    for pid in catalog:
        base = rng.uniform(40, 400)
        trend = rng.uniform(-3, 5)
        series = []
        level = base
        for w in range(weeks):
            noise = rng.gauss(0, base * 0.08)
            seasonal = 1.0 + 0.15 * (1 if (w % 4 == 3) else 0)  # mild monthly bump
            value = max(0.0, (level + trend * w) * seasonal + noise)
            series.append(round(value, 1))
        history[pid] = series
    return history


def generate_current_inventory(catalog: Dict[str, Dict[str, Any]], seed: int = 11) -> Dict[str, int]:
    rng = random.Random(seed)
    return {pid: rng.randint(0, 400) for pid in catalog}


def generate_suppliers(catalog: Dict[str, Dict[str, Any]], seed: int = 13) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rng = random.Random(seed)
    suppliers = {}
    for pid, info in catalog.items():
        n_candidates = rng.randint(2, 4)
        candidates = {}
        for j in range(n_candidates):
            sname = f"{pid}-SUP-{j+1}"
            candidates[sname] = {
                "unit_price": round(info["unit_cost"] * rng.uniform(0.85, 1.05), 2),
                "lead_time_days": rng.randint(3, 21),
                "quality_score": round(rng.uniform(6.0, 9.8), 2),
                "reliability_score": round(rng.uniform(0.7, 0.99), 3),
            }
        suppliers[pid] = candidates
    return suppliers


def generate_transactions(catalog: Dict[str, Dict[str, Any]], n_transactions: int = 300,
                           fraud_fraction: float = 0.06, seed: int = 21
                           ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Returns (transactions, ground_truth_fraud_ids) — the ground truth
    is used only for offline evaluation, never fed to the agent."""
    rng = random.Random(seed)
    customers = [f"CUST-{i:04d}" for i in range(60)]
    products = list(catalog.keys())
    transactions = []
    ground_truth_fraud: List[str] = []

    base_ts = 1_700_000_000
    for i in range(n_transactions):
        tid = f"TXN-{i:05d}"
        cust = rng.choice(customers)
        pid = rng.choice(products)
        price = catalog[pid]["current_price"]
        qty = rng.randint(1, 4)
        amount = round(price * qty, 2)
        country = rng.choice(COUNTRIES)
        billing = country
        shipping = country
        ts = base_ts + i * rng.randint(30, 900)

        is_fraud = rng.random() < fraud_fraction
        if is_fraud:
            # inject real anomalous patterns: extreme amount + geo mismatch
            amount = round(amount * rng.uniform(4, 9), 2)
            shipping = rng.choice([c for c in COUNTRIES if c != billing])
            ground_truth_fraud.append(tid)

        transactions.append({
            "transaction_id": tid, "customer_id": cust, "product_id": pid,
            "amount": amount, "billing_country": billing, "shipping_country": shipping,
            "timestamp": ts,
        })

    # inject velocity-anomaly bursts for a few customers (also fraud)
    burst_customers = rng.sample(customers, 3)
    for bc in burst_customers:
        burst_ts = base_ts + rng.randint(0, 200_000)
        for k in range(5):
            tid = f"TXN-BURST-{bc}-{k}"
            pid = rng.choice(products)
            transactions.append({
                "transaction_id": tid, "customer_id": bc, "product_id": pid,
                "amount": round(catalog[pid]["current_price"], 2),
                "billing_country": "US", "shipping_country": "US",
                "timestamp": burst_ts + k * 60,
            })
            ground_truth_fraud.append(tid)

    return transactions, ground_truth_fraud


def build_full_dataset(seed: int = 42) -> Dict[str, Any]:
    catalog = generate_catalog(seed=seed)
    sales_history = generate_sales_history(catalog, seed=seed + 1)
    current_inventory = generate_current_inventory(catalog, seed=seed + 2)
    suppliers = generate_suppliers(catalog, seed=seed + 3)
    transactions, fraud_ground_truth = generate_transactions(catalog, seed=seed + 4)
    return {
        "catalog": catalog,
        "sales_history": sales_history,
        "current_inventory": current_inventory,
        "suppliers": suppliers,
        "transactions": transactions,
        "fraud_ground_truth": fraud_ground_truth,
    }
