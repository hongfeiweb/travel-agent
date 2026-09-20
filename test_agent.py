"""
Offline sanity check for the ReAct loop, using MockLLM so it runs with
no API key and no network. This exercises: multi-step tool use,
observation feed-back, and the Finish action.

Run: python3 test_agent.py
"""

from react_agent import ReActAgent, MockLLM
from tools import TOOLS, TOOL_DESCRIPTIONS


def test_two_step_calculation():
    scripted = [
        "Thought: I need to compute 12 * 7 first.\nAction: Calculator[12*7]",
        "Thought: Now add 5 to that result.\nAction: Calculator[84+5]",
        "Thought: I now know the final answer\nAction: Finish[89]",
    ]
    agent = ReActAgent(llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS, verbose=True)
    result = agent.run("What is (12 * 7) + 5?")

    assert result.answer == "89", f"expected '89', got {result.answer!r}"
    assert len(result.trace) == 3
    assert result.trace[0].observation == "84"
    assert result.trace[1].observation == "89"
    print("\n[PASS] test_two_step_calculation")


def test_search_tool():
    scripted = [
        "Thought: I should look up where Cézanne was born.\nAction: Search[cezanne birthplace]",
        "Thought: I now know the final answer\nAction: Finish[Aix-en-Provence, France]",
    ]
    agent = ReActAgent(llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS, verbose=True)
    result = agent.run("Where was Cézanne born?")

    assert "Aix-en-Provence" in result.answer
    assert "1839" in result.trace[0].observation
    print("\n[PASS] test_search_tool")


def test_unknown_tool_is_reported_not_crashed():
    scripted = [
        "Thought: let me try a tool that doesn't exist.\nAction: Teleport[Paris]",
        "Thought: that failed, I'll just answer directly.\nAction: Finish[I cannot teleport, but Paris is the capital of France]",
    ]
    agent = ReActAgent(llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS, verbose=True)
    result = agent.run("Teleport me to Paris")

    assert "unknown tool" in result.trace[0].observation.lower()
    assert result.answer is not None
    print("\n[PASS] test_unknown_tool_is_reported_not_crashed")


def test_step_cap_is_respected():
    # Script never finishes -> agent must stop at max_steps rather than loop forever.
    scripted = ["Thought: still thinking.\nAction: Calculator[1+1]"] * 10
    agent = ReActAgent(llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS, max_steps=3, verbose=False)
    result = agent.run("Never-ending question")

    assert result.hit_step_cap is True
    assert result.answer is None
    assert len(result.trace) == 3
    print("[PASS] test_step_cap_is_respected")


if __name__ == "__main__":
    test_two_step_calculation()
    test_search_tool()
    test_unknown_tool_is_reported_not_crashed()
    test_step_cap_is_respected()
    print("\nAll tests passed.")
