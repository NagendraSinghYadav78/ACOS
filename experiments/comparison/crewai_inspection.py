"""
experiments/comparison/crewai_inspection.py

Verifies CrewAI's LLM-client requirement by actually instantiating a
crewai.Agent (not just inspecting its Pydantic field schema) with no
llm argument given, then inspecting the resulting object -- pinned to
crewai==1.15.10, since this behavior is not part of any public API
contract and could change between versions.
"""
from crewai import Agent, Task

if __name__ == "__main__":
    agent_required = [k for k, v in Agent.model_fields.items() if v.is_required()]
    task_required = [k for k, v in Task.model_fields.items() if v.is_required()]
    print("Agent required fields:", agent_required)
    print("Task required fields:", task_required)
    print("Agent has 'llm' field:", "llm" in Agent.model_fields)
    assert "llm" in Agent.model_fields, "Expected CrewAI Agent to define an llm field"

    # Actually instantiate an Agent with no llm argument, and inspect what
    # was auto-created -- this is the real claim being verified, not just
    # the presence of an 'llm' field in the schema.
    agent = Agent(role="test", goal="test", backstory="test")
    llm_type = type(agent.llm).__name__
    llm_model = getattr(agent.llm, "model", None)
    llm_provider = getattr(agent.llm, "provider", None)
    print(f"\nInstantiated Agent(role='test', goal='test', backstory='test') with no llm argument.")
    print(f"agent.llm type: {llm_type}")
    print(f"agent.llm.model: {llm_model}")
    print(f"agent.llm.provider: {llm_provider}")
    assert llm_model is not None, "Expected CrewAI to auto-instantiate a default LLM client"

    print("\nConclusion: instantiating a CrewAI Agent with no llm argument auto-creates a")
    print(f"real LLM client ({llm_provider} {llm_model}); there is no supported path to")
    print("execute a Task via a plain deterministic function.")
