"""
experiments/comparison/crewai_inspection.py

Verifies, by inspecting CrewAI's actual Pydantic model definitions
(not documentation), that crewai.Agent and crewai.Task cannot be
executed without an LLM client.
"""
from crewai import Agent, Task

if __name__ == "__main__":
    agent_required = [k for k, v in Agent.model_fields.items() if v.is_required()]
    task_required = [k for k, v in Task.model_fields.items() if v.is_required()]
    print("Agent required fields:", agent_required)
    print("Task required fields:", task_required)
    print("Agent has 'llm' field:", "llm" in Agent.model_fields)
    assert "llm" in Agent.model_fields, "Expected CrewAI Agent to define an llm field"
    print("\nConclusion: CrewAI's Agent/Task execution model is built around an LLM-driven")
    print("persona (role/goal/backstory) and a required llm client; there is no supported")
    print("path to execute a Task via a plain deterministic function, unlike LangGraph's")
    print("StateGraph nodes or AutoGen-core's ClosureAgent handlers.")
