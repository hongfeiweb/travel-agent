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
  2. A parser that pulls the first "Action: Name[input]" out of the
     model's text, matching brackets by depth so a bracketed input
     doesn't get confused with the end of the action.
  3. A loop that: calls the LLM -> parses the action -> runs the tool
     -> appends the observation -> repeats, until Finish or a step cap.
     Output that doesn't parse gets re-prompted rather than ending the
     run.

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

ACTION_HEAD_RE = re.compile(r"Action:\s*(\w+)\[")


@dataclass
class ParsedStep:
    thought: Optional[str]
    action_name: Optional[str]
    action_input: Optional[str]
    raw_text: str


def _extract_bracketed(text: str, open_idx: int) -> Optional[str]:
    """
    Return the contents of the bracket that opens at text[open_idx].

    Scans forward tracking nesting depth rather than regex-matching, so
    that both of these parse the way a reader expects:

        Search[list[0] stuff]     -> "list[0] stuff"
        Calculator[1+1] followed
        by a second Action line   -> "1+1", not everything up to the
                                     final ']' in the whole response

    A greedy regex gets the first right and the second wrong; a non-greedy
    one gets the second right and the first wrong. Returns None if the
    bracket is never closed (e.g. a response truncated mid-action), which
    the caller treats as a parse failure rather than guessing at the
    intended input.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return None


def parse_step(text: str) -> ParsedStep:
    """Pull the Thought and the first Action[...] out of model output."""
    thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\Z)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None

    action_name = None
    action_input = None
    head = ACTION_HEAD_RE.search(text)
    if head:
        inner = _extract_bracketed(text, head.end() - 1)
        if inner is not None:
            action_name = head.group(1)
            action_input = inner.strip()

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
        max_parse_retries: int = 2,
        verbose: bool = True,
        memory_limit: int = 3,
        recent_turns_limit: int = 2,
        memory_mode: str = "summary",
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = build_system_prompt(tool_descriptions)
        self.max_steps = max_steps
        # Consecutive unparseable responses tolerated before giving up. Each
        # retry still costs a step, so max_steps stays the hard cap on LLM
        # calls. 0 restores the old behaviour of bailing on first failure.
        self.max_parse_retries = max_parse_retries
        self.verbose = verbose
        self.memory_limit = memory_limit
        self.recent_turns_limit = recent_turns_limit
        self.memory_mode = memory_mode.lower()
        if self.memory_mode not in {"full", "compact", "summary"}:
            raise ValueError("memory_mode must be one of: 'full', 'compact', 'summary'")

    def _format_recent_turns(self, recent_turns: List[Dict[str, str]]) -> str:
        if not recent_turns:
            return "(no prior reasoning steps)"
        compact = []
        for turn in recent_turns[-self.recent_turns_limit:]:
            role = turn["role"].title()
            compact.append(f"{role}: {turn['content']}")
        return "\n".join(compact)

    def _summarize_memory(self, memory: str) -> str:
        if not memory:
            return "No facts gathered yet."
        facts = [item.strip() for item in memory.split(" | ") if item.strip()]
        if len(facts) <= self.memory_limit:
            return "; ".join(facts)
        return "; ".join(facts[-self.memory_limit:])

    def _build_prompt_messages(self, question: str, memory: str, recent_turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.memory_mode == "full":
            return [{"role": "user", "content": f"Question: {question}"}] + recent_turns

        if self.memory_mode == "compact":
            recent_text = self._format_recent_turns(recent_turns)
            prompt = (
                f"Question: {question}\n"
                f"Memory: {self._summarize_memory(memory)}\n"
                f"Recent steps:\n{recent_text}"
            )
            return [{"role": "user", "content": prompt}]

        recent_text = self._format_recent_turns(recent_turns)
        history_summary = self._summarize_old_history(recent_turns)
        prompt = (
            f"Question: {question}\n"
            f"Memory: {self._summarize_memory(memory)}\n"
            f"History summary: {history_summary}\n"
            f"Recent steps:\n{recent_text}"
        )
        return [{"role": "user", "content": prompt}]

    def _update_memory(self, memory: str, parsed: ParsedStep, observation: str) -> str:
        if parsed.action_name == "Finish":
            return memory
        if parsed.action_name is None:
            return memory

        fact = f"{parsed.action_name}[{parsed.action_input}] -> {observation}"
        facts = [item.strip() for item in memory.split(" | ") if item.strip()] if memory else []
        facts.append(fact)
        if len(facts) > self.memory_limit:
            facts = facts[-self.memory_limit:]
        return " | ".join(facts)

    def _summarize_old_history(self, recent_turns: List[Dict[str, str]]) -> str:
        if not recent_turns:
            return "(no prior reasoning steps)"

        summary_items = []
        for turn in recent_turns:
            content = turn["content"].strip()
            if not content:
                continue
            summary_items.append(content)

        if not summary_items:
            return "(no prior reasoning steps)"
        return " | ".join(summary_items[-self.recent_turns_limit * 2:])

    def run(self, question: str) -> ReActResult:
        messages = [{"role": "user", "content": f"Question: {question}"}]
        recent_turns: List[Dict[str, str]] = []
        memory = ""
        trace: List[TraceStep] = []
        consecutive_parse_failures = 0

        for step_num in range(1, self.max_steps + 1):
            if self.memory_mode != "full":
                messages = self._build_prompt_messages(question, memory, recent_turns)
            raw = self.llm(self.system_prompt, messages)
            parsed = parse_step(raw)

            if self.verbose:
                print(f"\n--- Step {step_num} ---")
                if parsed.thought:
                    print(f"Thought: {parsed.thought}")
                if parsed.action_name:
                    print(f"Action: {parsed.action_name}[{parsed.action_input}]")

            if self.memory_mode != "full":
                recent_turns.append({"role": "assistant", "content": raw})
            else:
                messages.append({"role": "assistant", "content": raw})

            if parsed.action_name is None:
                # Model didn't follow the format. Tell it so and let it retry,
                # rather than throwing away a run over one malformed turn.
                consecutive_parse_failures += 1
                if consecutive_parse_failures > self.max_parse_retries:
                    if self.verbose:
                        print("No valid action after "
                              f"{self.max_parse_retries} retries — giving up.")
                    trace.append(TraceStep(parsed.thought, None, None, None))
                    return ReActResult(answer=None, trace=trace)

                reprompt = (
                    "Your last response did not contain a valid action. Reply with "
                    "a single Thought line followed by a single Action line of the "
                    "form 'Action: ToolName[input]', or 'Action: Finish[your answer]' "
                    f"if you are done. Available tools: {list(self.tools)}."
                )
                if self.verbose:
                    print(f"Observation: {reprompt}")
                trace.append(TraceStep(parsed.thought, None, None, reprompt))
                if self.memory_mode != "full":
                    recent_turns.append({"role": "user", "content": reprompt})
                    if len(recent_turns) > self.recent_turns_limit * 3:
                        recent_turns = recent_turns[-self.recent_turns_limit * 3:]
                else:
                    messages.append({"role": "user", "content": reprompt})
                continue

            consecutive_parse_failures = 0

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
            if self.memory_mode != "full":
                recent_turns.append({"role": "user", "content": f"Observation: {observation}"})
                if len(recent_turns) > self.recent_turns_limit * 3:
                    recent_turns = recent_turns[-self.recent_turns_limit * 3:]
                memory = self._update_memory(memory, parsed, observation)
            else:
                messages.append({"role": "user", "content": f"Observation: {observation}"})

        return ReActResult(answer=None, trace=trace, hit_step_cap=True)
