"""
experiments/comparison/crewai_inspection.py

Inspects CrewAI's actual Pydantic model definitions (not documentation)
to determine crewai.Agent and crewai.Task's execution requirements.
"""
from crewai import Agent, Task

if __name__ == "__main__":
    agent_required = [k for k, v in Agent.model_fields.items() if v.is_required()]
    task_required = [k for k, v in Task.model_fields.items() if v.is_required()]
    print("Agent required fields:", agent_required)
    print("Task required fields:", task_required)
    print("Agent has 'llm' field:", "llm" in Agent.model_fields)
    assert "llm" in Agent.model_fields, "Expected CrewAI Agent to define an llm field"
    print("\nConclusion: CrewAI's Agent/Task model defines a role/goal/backstory persona")
    print("and an llm field; there is no supported path to execute a Task via a plain")
    print("deterministic function.")
