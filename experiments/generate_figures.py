"""
experiments/generate_figures.py

Generates publication-quality figures FROM experiments/results.json,
i.e. from real measurements produced by run_experiments.py. Run
run_experiments.py first. Figures are saved as .png at 300 DPI into
figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from data.synthetic_data import build_full_dataset
from main import build_knowledge_graph

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def fig_scalability(results):
    e1 = results["E1_scalability_vs_catalog_size"]
    n = [r["n_products"] for r in e1]
    t = [r["wall_clock_seconds"] * 1000 for r in e1]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(n, t, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Catalog size (number of SKUs)")
    ax.set_ylabel("End-to-end workflow latency (ms)")
    ax.set_title("ACOS Workflow Latency vs. Catalog Size\n(single-process, no parallelism, measured locally)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scalability.png")
    plt.close(fig)


def fig_agent_latency_breakdown(results):
    e2 = results["E2_per_agent_latency_ms"]
    agents = list(e2.keys())
    latencies = [e2[a] for a in agents]
    order = np.argsort(latencies)[::-1]
    agents = [agents[i] for i in order]
    latencies = [latencies[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(agents, latencies, color="#0891b2")
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Per-Agent Latency Breakdown\n(single workflow run, measured)")
    ax.invert_yaxis()
    for bar, val in zip(bars, latencies):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.2f}",
                 va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_agent_latency.png")
    plt.close(fig)


def fig_fraud_confusion_matrix(results):
    e3 = results["E3_fraud_detection_quality"]
    matrix = np.array([[e3["true_positives"], e3["false_negatives"]],
                        [e3["false_positives"], e3["true_negatives"]]])
    labels = ["Predicted Fraud", "Predicted Legitimate"]
    row_labels = ["Actual Fraud", "Actual Legitimate"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(row_labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    ax.set_title(f"Fraud Screening Confusion Matrix (synthetic, seeded)\n"
                 f"Precision={e3['precision']:.2f}  Recall={e3['recall']:.2f}  F1={e3['f1']:.2f}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fraud_confusion_matrix.png")
    plt.close(fig)


def fig_forecast_accuracy_distribution():
    """Recomputes and plots the per-SKU in-sample MAPE distribution
    directly (not just the summary stats) for a richer figure."""
    orchestrator, dataset, _, ltm, _ = build_system_local()
    inputs = {"sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
              "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
              "transactions": dataset["transactions"]}
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    ltm.close()
    forecasts = result.task_outputs["forecast_demand"]["forecasts"]
    mapes = sorted(f["in_sample_mape"] for f in forecasts.values() if "in_sample_mape" in f)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(len(mapes)), mapes, color="#7c3aed")
    ax.axhline(np.mean(mapes), color="black", linestyle="--", linewidth=1,
               label=f"mean={np.mean(mapes):.3f}")
    ax.set_xlabel("SKU (sorted by MAPE)")
    ax.set_ylabel("In-sample MAPE (Holt's linear smoothing)")
    ax.set_title("Demand Forecast In-Sample Error by SKU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_forecast_mape.png")
    plt.close(fig)


def build_system_local():
    from main import build_system
    return build_system(db_path="/tmp/acos_figs.db")


def fig_governance_sensitivity(results):
    e5 = results["E5_price_search_bound_vs_escalation_rate"]
    x = [r["max_price_change_pct_allowed"] * 100 for r in e5]
    y = [r["n_escalated"] for r in e5]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(x, y, marker="s", color="#dc2626", linewidth=2)
    ax.axvline(20, color="gray", linestyle=":", label="Policy escalation threshold (20%)")
    ax.set_xlabel("PricingAgent max allowed price change (%)")
    ax.set_ylabel("Number of pricing decisions escalated")
    ax.set_title("Governance Escalations vs. Agent Search Bound\n(fixed policy thresholds, measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_governance_sensitivity.png")
    plt.close(fig)


def fig_latency_stability(results):
    e6 = results["E6_latency_stability"]
    runs = e6["raw_seconds"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(1, len(runs) + 1), [t * 1000 for t in runs], marker="o", color="#059669")
    ax.axhline(e6["mean_seconds"] * 1000, color="black", linestyle="--",
               label=f"mean={e6['mean_seconds']*1000:.1f}ms, std={e6['stdev_seconds']*1000:.1f}ms")
    ax.set_xlabel("Run number")
    ax.set_ylabel("Workflow latency (ms)")
    ax.set_title(f"Latency Stability Across {e6['n_runs']} Repeated Runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_latency_stability.png")
    plt.close(fig)


def fig_architecture_diagram():
    """A schematic system architecture diagram (structural, not data-driven)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")

    layers = [
        ("API Layer (FastAPI / REST / OpenAPI)", "#bfdbfe"),
        ("Orchestrator  +  Planner  +  Consensus Resolver", "#a7f3d0"),
        ("Agent Layer: Forecast | Inventory | Pricing | Procurement | Fraud | Analytics | Governance", "#fbcfe8"),
        ("Shared Memory (blackboard)  |  Event Bus (pub/sub)", "#ddd6fe"),
        ("Long-Term Memory (SQLite)  |  Vector Memory (TF-IDF+FAISS)  |  Knowledge Graph (networkx)", "#fed7aa"),
        ("Policy / Governance Engine  |  Monitoring & Audit Log", "#fca5a5"),
    ]
    y = 0.95
    h = 0.12
    for label, color in layers:
        ax.add_patch(plt.Rectangle((0.05, y - h), 0.9, h * 0.85, facecolor=color,
                                    edgecolor="black", linewidth=1.2))
        ax.text(0.5, y - h / 2, label, ha="center", va="center", fontsize=9, wrap=True)
        y -= h
    ax.set_title("ACOS Layered System Architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_architecture.png")
    plt.close(fig)


def fig_knowledge_graph():
    dataset = build_full_dataset(seed=42)
    kg = build_knowledge_graph(dataset)
    g = kg.g

    # Subsample for a readable figure: a handful of products, at most 2
    # suppliers each (full graph has 90 nodes / 70 edges -- too dense to
    # render legibly at print size).
    products = [n for n, d in g.nodes(data=True) if d.get("type") == "Product"][:5]
    suppliers = set()
    for p in products:
        suppliers.update(kg.suppliers_for(p)[:2])
    sub_nodes = set(products) | suppliers
    sub = g.subgraph(sub_nodes)

    pos = nx.spring_layout(sub, seed=7, k=1.3)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    product_nodes = [n for n in sub.nodes if sub.nodes[n]["type"] == "Product"]
    supplier_nodes = [n for n in sub.nodes if sub.nodes[n]["type"] == "Supplier"]
    nx.draw_networkx_nodes(sub, pos, nodelist=product_nodes, node_color="#60a5fa",
                            node_size=500, label="Product", ax=ax)
    nx.draw_networkx_nodes(sub, pos, nodelist=supplier_nodes, node_color="#f97316",
                            node_size=350, label="Supplier", ax=ax)
    nx.draw_networkx_edges(sub, pos, alpha=0.4, arrows=True, ax=ax)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
    ax.legend(scatterpoints=1)
    ax.set_title("Enterprise Knowledge Graph (sampled subgraph)\nSUPPLIES edges: Supplier -> Product")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_knowledge_graph.png")
    plt.close(fig)


def main():
    results = load_results()
    fig_scalability(results)
    fig_agent_latency_breakdown(results)
    fig_fraud_confusion_matrix(results)
    fig_forecast_accuracy_distribution()
    fig_governance_sensitivity(results)
    fig_latency_stability(results)
    fig_architecture_diagram()
    fig_knowledge_graph()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
