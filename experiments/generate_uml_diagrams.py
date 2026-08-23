"""
experiments/generate_uml_diagrams.py

Generates UML-style diagrams (component, sequence, class, deployment,
agent lifecycle) using Graphviz, derived directly from the actual class
and module structure in this repository (not invented). Run after
generate_figures.py. Outputs to figures/.
"""
import graphviz
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

COMMON_ATTRS = dict(fontname="Helvetica", fontsize="11")


def render(g: graphviz.Digraph, name: str):
    out = g.render(filename=str(FIG_DIR / name), format="png", cleanup=True)
    print("wrote", out)


# ---------------------------------------------------------------- COMPONENT
def component_diagram():
    g = graphviz.Digraph("component", graph_attr={"rankdir": "TB", "splines": "ortho", "fontname": "Helvetica"})
    g.attr("node", shape="component", style="filled", fillcolor="#e0f2fe", fontname="Helvetica", fontsize="10")

    g.node("api", "api/main.py\n(FastAPI service)")
    g.node("orch", "core/orchestrator.py\nOrchestrator")
    g.node("planner", "core/planner.py\nPlanner")
    g.node("consensus", "core/consensus.py\nConsensusResolver")
    g.node("bus", "core/event_bus.py\nEventBus")
    g.node("smem", "core/memory.py\nSharedMemory")
    g.node("ltm", "core/memory.py\nLongTermMemory\n(SQLite)")
    g.node("vmem", "core/vector_memory.py\nVectorMemory\n(TF-IDF+FAISS)")
    g.node("kg", "core/knowledge_graph.py\nKnowledgeGraph\n(networkx)")
    g.node("policy", "core/policy.py\nPolicyEngine")
    g.node("agents", "agents/*.py\n7 specialized agents\n(subclass BaseAgent)", fillcolor="#fce7f3")

    g.edge("api", "orch", label="run_workflow()")
    g.edge("orch", "planner", label="build_plan()\nschedule()")
    g.edge("orch", "agents", label="dispatch tasks")
    g.edge("agents", "smem", label="perceive()/act()")
    g.edge("agents", "bus", label="publish()")
    g.edge("agents", "ltm", label="record_episode()\nrecord_decision()")
    g.edge("orch", "consensus", label="reconcile_price_and_inventory()")
    g.edge("agents", "policy", label="evaluate()", style="dashed", constraint="false")
    g.edge("agents", "kg", label="query", style="dashed", constraint="false")
    g.edge("agents", "vmem", label="search()", style="dashed", constraint="false")

    render(g, "fig_uml_component")


# ---------------------------------------------------------------- SEQUENCE
def sequence_diagram():
    g = graphviz.Digraph("sequence", graph_attr={"rankdir": "LR", "fontname": "Helvetica"})
    g.attr("node", shape="box", style="filled", fillcolor="#f1f5f9", fontname="Helvetica", fontsize="10")

    participants = ["Client", "API", "Orchestrator", "Planner", "Agent(s)", "SharedMemory", "ConsensusResolver", "GovernanceAgent", "LongTermMemory"]
    with g.subgraph(name="cluster_lifeline") as c:
        c.attr(rank="same")
        for p in participants:
            c.node(p, p)

    # Represent the call sequence as a simple numbered chain (Graphviz doesn't
    # do true UML lifelines well without a dedicated library; we approximate
    # with a left-to-right numbered call graph, which is accurate to the
    # real call order in core/orchestrator.py::run_workflow).
    steps = [
        ("Client", "API", "1: POST /workflows/run"),
        ("API", "Orchestrator", "2: run_workflow(goal, inputs)"),
        ("Orchestrator", "Planner", "3: build_plan() + schedule()"),
        ("Orchestrator", "Agent(s)", "4: agent.run(task_id, ctx) per wave"),
        ("Agent(s)", "SharedMemory", "5: perceive() / act()"),
        ("Agent(s)", "LongTermMemory", "6: record_episode()"),
        ("Orchestrator", "ConsensusResolver", "7: reconcile_price_and_inventory()"),
        ("Orchestrator", "GovernanceAgent", "8: final task in DAG: review()"),
        ("GovernanceAgent", "LongTermMemory", "9: record_decision() per ruling"),
        ("Orchestrator", "API", "10: return WorkflowResult"),
        ("API", "Client", "11: JSON response"),
    ]
    for i, (src, dst, label) in enumerate(steps):
        g.edge(src, dst, label=f" {label}", fontsize="9")

    render(g, "fig_uml_sequence")


# ---------------------------------------------------------------- CLASS
def class_diagram():
    g = graphviz.Digraph("class", graph_attr={"rankdir": "BT", "fontname": "Helvetica"})
    g.attr("node", shape="record", fontname="Helvetica", fontsize="10", style="filled", fillcolor="#fefce8")

    g.node("BaseAgent", "{BaseAgent|+ name: str|+ event_bus: EventBus|+ shared_memory: SharedMemory|"
                        "+ long_term_memory: LongTermMemory|"
                        "+ perceive(context) : dict|+ reason(context) : Decision|"
                        "+ reflect(decision) : Decision|+ act(task_id, decision) : Decision|"
                        "+ run(task_id, context) : Decision}")

    subclasses = [
        ("DemandForecastAgent", "holt_linear_forecast()"),
        ("InventoryAgent", "EOQ / reorder point"),
        ("PricingAgent", "elasticity optimization"),
        ("ProcurementAgent", "TOPSIS ranking"),
        ("FraudRiskAgent", "robust z-score ensemble"),
        ("AnalyticsAgent", "KPI rollup"),
        ("GovernanceAgent", "PolicyEngine.evaluate()"),
    ]
    for name, method in subclasses:
        g.node(name, f"{{{name}|+ reason(context) : Decision\\l  ({method})\\l}}")
        g.edge(name, "BaseAgent", arrowhead="empty")

    g.node("Decision", "{Decision|+ action: str|+ output: dict|+ confidence: float|"
                       "+ rationale: str|+ warnings: list}")
    g.edge("BaseAgent", "Decision", label="produces", style="dashed", arrowhead="vee")

    g.node("Orchestrator", "{Orchestrator|+ agents: dict[str, BaseAgent]|+ planner: Planner|"
                           "+ consensus: ConsensusResolver|+ run_workflow(goal, inputs) : WorkflowResult}")
    g.edge("Orchestrator", "BaseAgent", label="dispatches to", style="dashed", arrowhead="vee")

    render(g, "fig_uml_class")


# ---------------------------------------------------------------- DEPLOYMENT
def deployment_diagram():
    g = graphviz.Digraph("deployment", graph_attr={"rankdir": "LR", "fontname": "Helvetica"})
    g.attr("node", shape="box3d", style="filled", fillcolor="#ecfccb", fontname="Helvetica", fontsize="10")

    with g.subgraph(name="cluster_process") as c:
        c.attr(label="Single Python process (this implementation)", style="dashed", fontname="Helvetica", fontsize="10")
        c.node("uvicorn", "uvicorn / FastAPI\n(api/main.py)")
        c.node("orch2", "Orchestrator + Agents\n(in-process)")
        c.node("sqlite", "SQLite file\n(long-term memory)", shape="cylinder", fillcolor="#fde68a")
        c.node("faiss_local", "FAISS index\n(in-process)", shape="cylinder", fillcolor="#fde68a")
        c.node("nx", "networkx graph\n(in-process)", shape="cylinder", fillcolor="#fde68a")
        c.edge("uvicorn", "orch2")
        c.edge("orch2", "sqlite")
        c.edge("orch2", "faiss_local")
        c.edge("orch2", "nx")

    g.node("client_ext", "Client\n(browser / curl / tests)", shape="box", fillcolor="#dbeafe")
    g.edge("client_ext", "uvicorn", label="HTTP + bearer token")

    with g.subgraph(name="cluster_prod") as c:
        c.attr(label="Production target (not deployed here)",
               style="dashed", color="gray50", fontname="Helvetica", fontsize="10")
        c.node("neo4j", "Neo4j cluster", shape="box3d", fillcolor="#e5e7eb")
        c.node("postgres", "PostgreSQL", shape="box3d", fillcolor="#e5e7eb")
        c.node("qdrant", "Qdrant / vector DB", shape="box3d", fillcolor="#e5e7eb")
        c.node("k8s", "Kubernetes\n(horizontal agent scaling)", shape="box3d", fillcolor="#e5e7eb")

    render(g, "fig_uml_deployment")


# ---------------------------------------------------------------- LIFECYCLE
def lifecycle_diagram():
    g = graphviz.Digraph("lifecycle", graph_attr={"rankdir": "LR", "fontname": "Helvetica"})
    g.attr("node", shape="circle", style="filled", fillcolor="#ede9fe", fontname="Helvetica", fontsize="10", width="1.1")

    states = ["Dispatched", "Perceiving", "Reasoning", "Reflecting", "Acting", "Completed"]
    for s in states:
        g.node(s, s)
    g.edge("Dispatched", "Perceiving", label="perceive(context)")
    g.edge("Perceiving", "Reasoning", label="reason(context)")
    g.edge("Reasoning", "Reflecting", label="reflect(decision)")
    g.edge("Reflecting", "Acting", label="act(task_id, decision)")
    g.edge("Acting", "Completed", label="publish event +\npersist episode")
    g.edge("Reflecting", "Reasoning", label="confidence<0.3:\nflagged, not re-run\n(logged only)", style="dashed", constraint="false")

    render(g, "fig_uml_lifecycle")


if __name__ == "__main__":
    component_diagram()
    sequence_diagram()
    class_diagram()
    deployment_diagram()
    lifecycle_diagram()
    print("All UML-style diagrams generated.")
