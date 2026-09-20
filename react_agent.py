"""
ReAct from scratch.

ReAct (Yao et al., 2022, "ReAct: Synergizing Reasoning and Acting in
Language Models") interleaves three things in a single running
transcript that gets fed back into the LLM on every step:

    Thought: <the model reasons about what to do next>
    Action: ToolName[tool input]
    Observation: <the result of running the tool, appended by us>

    ... repeated ...

    Thought: I now know the answer
    Action: Finish[final answer]

There's no framework here — just:
  1. A prompt template that teaches the model this format.
  2. A regex-based parser that pulls "Action: Name[input]" out of the
     model's text.
  3. A loop that: calls the LLM -> parses the action -> runs the tool
     -> appends the observation -> repeats, until Finish or a step cap.

Two LLM backends are included:
  - AnthropicLLM: calls the real Claude API (needs ANTHROPIC_API_KEY).
  - MockLLM: a scripted, offline stand-in used by test_agent.py so you
    can see/step through the control flow with zero API calls.

Swap backends via the `llm` argument to ReActAgent — the agent loop
itself doesn't care which one it's talking to.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are an agent that answers questions by reasoning step by step \
and, when needed, using tools. Follow this exact format:

Thought: reason about what you know and what to do next
Action: ToolName[input to the tool]

You will then be given an Observation with the tool's result. Continue \
the Thought/Action/Observation cycle as needed. When you are confident \
in the final answer, respond with:

Thought: I now know the final answer
Action: Finish[your final answer here]

Rules:
- Only ever output ONE Thought and ONE Action per turn, then stop and \
wait for the Observation. Never write an Observation yourself.
- The Action line must always be one of the tools below, or Finish.

Available tools:
{tool_list}
"""


def build_system_prompt(tool_descriptions: Dict[str, str]) -> str:
    tool_list = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())
    return SYSTEM_PROMPT_TEMPLATE.format(tool_list=tool_list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*)\]", re.DOTALL)


@dataclass
class ParsedStep:
    thought: Optional[str]
    action_name: Optional[str]
    action_input: Optional[str]
    raw_text: str


def parse_step(text: str) -> ParsedStep:
    """Pull the Thought and the first Action[...] out of model output."""
    thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\Z)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None

    action_match = ACTION_RE.search(text)
    if action_match:
        action_name = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
    else:
        action_name = None
        action_input = None

    return ParsedStep(thought=thought, action_name=action_name, action_input=action_input, raw_text=text)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class AnthropicLLM:
    """Thin wrapper around the real Anthropic API."""

    def __init__(self, model: Optional[str] = None):
        import anthropic  # imported lazily so MockLLM works with no SDK/key

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model or os.environ.get("REACT_SANDBOX_MODEL", "claude-sonnet-5")

    def __call__(self, system: str, messages: List[Dict[str, str]]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=messages,
            stop_sequences=["\nObservation:"],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class MockLLM:
    """
    A scripted offline "LLM" for testing the agent loop without any API
    calls. Give it a list of canned responses; it returns them in order,
    regardless of what it's asked. Useful for verifying the parser and
    the loop mechanics deterministically.
    """

    def __init__(self, scripted_responses: List[str]):
        self._responses = list(scripted_responses)
        self._i = 0

    def __call__(self, system: str, messages: List[Dict[str, str]]) -> str:
        if self._i >= len(self._responses):
            raise RuntimeError("MockLLM ran out of scripted responses")
        resp = self._responses[self._i]
        self._i += 1
        return resp


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    thought: Optional[str]
    action_name: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]


@dataclass
class ReActResult:
    answer: Optional[str]
    trace: List[TraceStep] = field(default_factory=list)
    hit_step_cap: bool = False


class ReActAgent:
    def __init__(
        self,
        llm: Callable[[str, List[Dict[str, str]]], str],
        tools: Dict[str, Callable[[str], str]],
        tool_descriptions: Dict[str, str],
        max_steps: int = 6,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = build_system_prompt(tool_descriptions)
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> ReActResult:
        messages = [{"role": "user", "content": f"Question: {question}"}]
        trace: List[TraceStep] = []

        for step_num in range(1, self.max_steps + 1):
            raw = self.llm(self.system_prompt, messages)
            parsed = parse_step(raw)

            if self.verbose:
                print(f"\n--- Step {step_num} ---")
                if parsed.thought:
                    print(f"Thought: {parsed.thought}")
                if parsed.action_name:
                    print(f"Action: {parsed.action_name}[{parsed.action_input}]")

            # The model's own turn goes into the transcript as-is.
            messages.append({"role": "assistant", "content": raw})

            if parsed.action_name is None:
                # Model didn't follow the format — stop rather than loop blindly.
                trace.append(TraceStep(parsed.thought, None, None, None))
                return ReActResult(answer=None, trace=trace)

            if parsed.action_name == "Finish":
                trace.append(TraceStep(parsed.thought, "Finish", parsed.action_input, None))
                return ReActResult(answer=parsed.action_input, trace=trace)

            tool_fn = self.tools.get(parsed.action_name)
            if tool_fn is None:
                observation = f"Error: unknown tool '{parsed.action_name}'. Available: {list(self.tools)}"
            else:
                observation = tool_fn(parsed.action_input or "")

            if self.verbose:
                print(f"Observation: {observation}")

            trace.append(TraceStep(parsed.thought, parsed.action_name, parsed.action_input, observation))
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return ReActResult(answer=None, trace=trace, hit_step_cap=True)
