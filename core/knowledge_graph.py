"""
core/knowledge_graph.py

Product/supplier/customer relationship graph, backed by networkx
instead of a standalone Neo4j instance (no separate DB daemon in this
environment). Supports neighbor queries, path queries, and centrality-
based risk propagation -- roughly what a Cypher store would give you
at small-to-medium scale.

Node types: Product, Supplier, Customer, Warehouse, Category
Edge types: SUPPLIES, PURCHASED, SUBSTITUTES, LOCATED_AT, BELONGS_TO
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    # ---- construction -------------------------------------------------
    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        self.g.add_node(node_id, type=node_type, **attrs)

    def add_edge(self, src: str, dst: str, relation: str, **attrs: Any) -> None:
        self.g.add_edge(src, dst, key=relation, relation=relation, **attrs)

    # ---- queries --------------------------------------------------------
    def neighbors(self, node_id: str, relation: Optional[str] = None) -> List[str]:
        if node_id not in self.g:
            return []
        out = []
        for _, dst, data in self.g.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                out.append(dst)
        return out

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return dict(self.g.nodes[node_id]) if node_id in self.g else {}

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.g, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_substitutes(self, product_id: str) -> List[str]:
        return self.neighbors(product_id, relation="SUBSTITUTES")

    def suppliers_for(self, product_id: str) -> List[str]:
        # suppliers point to products via SUPPLIES edges
        return [n for n in self.g.predecessors(product_id)
                if any(d.get("relation") == "SUPPLIES"
                       for d in self.g.get_edge_data(n, product_id).values())]

    def supply_chain_risk_propagation(self, at_risk_supplier: str,
                                       damping: float = 0.6) -> Dict[str, float]:
        """Propagate a disruption risk score outward from a supplier node
        through SUPPLIES / SUBSTITUTES edges using a bounded BFS decay
        model: risk(node) = damping^(hop distance). Used by the
        Procurement/Risk agents to flag downstream exposure."""
        if at_risk_supplier not in self.g:
            return {}
        risk: Dict[str, float] = {at_risk_supplier: 1.0}
        frontier = [(at_risk_supplier, 1.0)]
        visited = {at_risk_supplier}
        while frontier:
            node, score = frontier.pop(0)
            for _, dst, data in self.g.out_edges(node, data=True):
                if dst in visited:
                    continue
                propagated = score * damping
                if propagated < 0.01:
                    continue
                risk[dst] = max(risk.get(dst, 0.0), propagated)
                visited.add(dst)
                frontier.append((dst, propagated))
        return risk

    def centrality_ranking(self, node_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """PageRank-based importance ranking, e.g. to identify critical
        suppliers whose disruption would have outsized downstream impact."""
        if self.g.number_of_nodes() == 0:
            return []
        pr = nx.pagerank(nx.DiGraph(self.g))
        items = [(n, s) for n, s in pr.items()
                 if node_type is None or self.g.nodes[n].get("type") == node_type]
        return sorted(items, key=lambda x: -x[1])

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}
