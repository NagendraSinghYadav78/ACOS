# Framework comparison scripts

These scripts produce an honest, executed-not-just-read structural
comparison against other multi-agent orchestration frameworks. They
are not part of ACOS itself — they exercise *other* frameworks
(LangGraph, AutoGen-core, CrewAI) to measure and inspect their actual
behavior rather than relying on documentation.

## Requirements

These need extra dependencies not in the main `requirements.txt`
(kept separate deliberately, since they are heavy and only needed for
this comparison, not to run ACOS itself):

```bash
pip install --break-system-packages langgraph pyautogen crewai
```

## Scripts

- `langgraph_equivalent_dag.py` — builds ACOS's 7-task DAG as a
  LangGraph `StateGraph` with no-op nodes; measures orchestration-only
  latency. Also documents a real `InvalidUpdateError` encountered and
  fixed during development (concurrent writes need an explicit reducer).
- `autogen_core_equivalent_dag.py` — same idea using AutoGen-core's
  `SingleThreadedAgentRuntime` and `ClosureAgent`; documents that this
  substrate has no native DAG/dependency abstraction (flat pub/sub only).
- `crewai_inspection.py` — inspects `crewai.Agent`/`crewai.Task`'s
  actual Pydantic field definitions to verify (not assume) that CrewAI's
  execution model requires an LLM client and cannot run a deterministic,
  credential-free task the way the other two frameworks can.
- `larger_dag_benchmark.py` — repeats the ACOS-vs-LangGraph
  scheduling-only latency comparison at 50 nodes (5 layers x 10 parallel
  nodes) instead of 7, since 7 nodes is too small to support any general
  claim about scheduler efficiency.

## Reproducing

```bash
python3 experiments/comparison/langgraph_equivalent_dag.py
python3 experiments/comparison/autogen_core_equivalent_dag.py
python3 experiments/comparison/crewai_inspection.py
python3 experiments/comparison/larger_dag_benchmark.py
```
