# Framework comparison scripts

**LangGraph, AutoGen-core, CrewAI, and Isolation Forest (scikit-learn) are
comparison-only experimental dependencies, not ACOS runtime components.**
ACOS itself installs and runs from the main `requirements.txt` alone.

These scripts install and execute each framework directly to measure
and inspect its actual behavior, rather than relying on documentation.

## Requirements

Extra dependencies not in the main `requirements.txt` (kept separate
since they are only needed for this comparison):

```bash
pip install --break-system-packages langgraph==1.2.10 pyautogen==0.10.0 crewai==1.15.10
```

## Scripts

- `langgraph_equivalent_dag.py` — builds ACOS's 7-task DAG as a
  LangGraph `StateGraph` with no-op nodes; measures orchestration-only
  latency.
- `autogen_core_equivalent_dag.py` — same idea using AutoGen-core's
  `SingleThreadedAgentRuntime` and `ClosureAgent`.
- `crewai_inspection.py` — inspects `crewai.Agent`/`crewai.Task`'s
  Pydantic field definitions directly to verify their runtime
  requirements.
- `larger_dag_benchmark.py` — repeats the ACOS-vs-LangGraph
  scheduling-only latency comparison at 50 nodes instead of 7.

## Reproducing

```bash
python3 experiments/comparison/langgraph_equivalent_dag.py
python3 experiments/comparison/autogen_core_equivalent_dag.py
python3 experiments/comparison/crewai_inspection.py
python3 experiments/comparison/larger_dag_benchmark.py
```
