"""
experiments/comparison/autogen_core_equivalent_dag.py

Exercises AutoGen-core's SingleThreadedAgentRuntime (the message-passing
substrate underneath autogen-agentchat) with one ClosureAgent handler
per ACOS task name, using plain Python closures (no LLM call). Note
this is a FLAT fan-out, not a dependency-ordered DAG: AutoGen-core's
pub/sub model has no native construct for "run assess_inventory only
after forecast_demand completes" the way LangGraph's edges or ACOS's
Planner.schedule() do. Expressing real dependency ordering here would
require each handler to explicitly publish a message to trigger the
next stage -- additional application code beyond what this script
does. That gap (no native dependency-DAG support in AutoGen-core) is
the real finding here, not an oversight in this script.

Run: python3 experiments/comparison/autogen_core_equivalent_dag.py
"""
import asyncio, time, statistics
from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime, ClosureAgent, ClosureContext, AgentId, TopicId, TypeSubscription

@dataclass
class TaskMessage:
    task_id: str
    payload: dict

results = {}

async def make_handler(task_id):
    async def handler(_agent: ClosureContext, message: TaskMessage, ctx) -> None:
        results[message.task_id] = {"ok": True}
    return handler

async def main():
    runtime = SingleThreadedAgentRuntime()
    task_ids = ["forecast_demand", "assess_inventory", "optimize_price", "select_supplier",
                "screen_transactions", "analytics_rollup", "governance_review"]

    for tid in task_ids:
        await ClosureAgent.register_closure(
            runtime, f"agent_{tid}", await make_handler(tid),
            subscriptions=lambda tid=tid: [TypeSubscription(topic_type=tid, agent_type=f"agent_{tid}")],
        )

    runtime.start()
    t0 = time.perf_counter()
    for tid in task_ids:
        await runtime.publish_message(TaskMessage(task_id=tid, payload={}), topic_id=TopicId(type=tid, source="orchestrator"))
    await runtime.stop_when_idle()
    elapsed = (time.perf_counter() - t0) * 1000
    print("AutoGen-core result keys:", list(results.keys()))
    print(f"AutoGen-core latency: {elapsed:.3f} ms")

asyncio.run(main())
