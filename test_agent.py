"""
Offline sanity check for the ReAct loop, using MockLLM so it runs with
no API key and no network. This exercises: multi-step tool use,
observation feed-back, and the Finish action.

Run: python3 test_agent.py
"""

from react_agent import ReActAgent, MockLLM, parse_step
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


def test_parser_stops_at_the_first_action():
    # A model that ignores the one-action-per-turn rule must not have its
    # first action's input swallow everything up to the last ']'.
    parsed = parse_step(
        "Thought: a\nAction: Calculator[12*7]\nThought: b\nAction: Finish[84]"
    )
    assert parsed.action_name == "Calculator"
    assert parsed.action_input == "12*7", f"got {parsed.action_input!r}"
    assert parsed.thought == "a"
    print("[PASS] test_parser_stops_at_the_first_action")


def test_parser_keeps_brackets_inside_the_input():
    parsed = parse_step("Thought: t\nAction: Search[list[0] stuff]")
    assert parsed.action_name == "Search"
    assert parsed.action_input == "list[0] stuff", f"got {parsed.action_input!r}"
    print("[PASS] test_parser_keeps_brackets_inside_the_input")


def test_parser_rejects_an_unclosed_action():
    # e.g. a response cut off by max_tokens mid-action. Better to report no
    # action (and re-prompt) than to hand back a truncated input.
    parsed = parse_step("Thought: t\nAction: Finish[the answer is 4")
    assert parsed.action_name is None
    print("[PASS] test_parser_rejects_an_unclosed_action")


def test_reprompt_recovers_from_unparseable_output():
    scripted = [
        "I reckon it's Aix-en-Provence.",  # no Action line at all
        "Thought: I should use the format.\nAction: Finish[Aix-en-Provence]",
    ]
    agent = ReActAgent(llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS, verbose=True)
    result = agent.run("Where was Cezanne born?")

    assert result.answer == "Aix-en-Provence", f"got {result.answer!r}"
    assert len(result.trace) == 2
    assert result.trace[0].action_name is None
    assert "did not contain a valid action" in result.trace[0].observation
    print("\n[PASS] test_reprompt_recovers_from_unparseable_output")


def test_reprompt_gives_up_after_the_retry_budget():
    scripted = ["waffling, no action here."] * 10
    agent = ReActAgent(
        llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS,
        max_steps=10, max_parse_retries=2, verbose=False,
    )
    result = agent.run("Unanswerable")

    assert result.answer is None
    assert result.hit_step_cap is False, "should stop on retries, not the step cap"
    assert len(result.trace) == 3, f"1 failure + 2 retries, got {len(result.trace)}"
    print("[PASS] test_reprompt_gives_up_after_the_retry_budget")


def test_parse_failure_counter_resets_on_a_good_turn():
    scripted = [
        "no action",
        "Thought: ok\nAction: Calculator[1+1]",
        "no action again",
        "Thought: done\nAction: Finish[2]",
    ]
    agent = ReActAgent(
        llm=MockLLM(scripted), tools=TOOLS, tool_descriptions=TOOL_DESCRIPTIONS,
        max_steps=10, max_parse_retries=1, verbose=False,
    )
    result = agent.run("What is 1+1?")

    # Without a reset, the second failure would exhaust the budget and bail.
    assert result.answer == "2", f"got {result.answer!r}"
    print("[PASS] test_parse_failure_counter_resets_on_a_good_turn")


if __name__ == "__main__":
    test_two_step_calculation()
    test_search_tool()
    test_unknown_tool_is_reported_not_crashed()
    test_step_cap_is_respected()
    test_parser_stops_at_the_first_action()
    test_parser_keeps_brackets_inside_the_input()
    test_parser_rejects_an_unclosed_action()
    test_reprompt_recovers_from_unparseable_output()
    test_reprompt_gives_up_after_the_retry_budget()
    test_parse_failure_counter_resets_on_a_good_turn()
    print("\nAll tests passed.")
