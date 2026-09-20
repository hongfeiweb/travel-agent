"""
Run the ReAct agent against the real Claude API.

Requires: pip install anthropic
Requires: export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python3 run_live.py "What is the Eiffel Tower's height in feet?"
"""

import sys

from react_agent import ReActAgent, AnthropicLLM
from tools import TOOLS, TOOL_DESCRIPTIONS


def main():
    if len(sys.argv) < 2:
        question = "Where was Van Gogh living when he painted Starry Night, and how far is that from Aix-en-Provence roughly?"
        print(f"No question given, using default:\n  {question}\n")
    else:
        question = " ".join(sys.argv[1:])

    agent = ReActAgent(
        llm=AnthropicLLM(),  # reads REACT_SANDBOX_MODEL env var, defaults to claude-sonnet-5
        tools=TOOLS,
        tool_descriptions=TOOL_DESCRIPTIONS,
        max_steps=6,
        verbose=True,
    )
    result = agent.run(question)

    print("\n=================")
    if result.answer is not None:
        print(f"FINAL ANSWER: {result.answer}")
    elif result.hit_step_cap:
        print("Agent hit the step cap without finishing.")
    else:
        print("Agent stopped without producing a parseable action.")
        print("Raw trace's last thought:", result.trace[-1].thought if result.trace else None)


if __name__ == "__main__":
    main()
