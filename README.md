# ReAct From Scratch

A minimal, dependency-light implementation of the **ReAct** pattern
(Yao et al., 2022) — no LangChain, no agent framework, just the prompt,
the parser, and the loop, so you can see exactly what's happening at
each step.

## Files

| File | Purpose |
|---|---|
| `react_agent.py` | The core loop: prompt template, `Action: Name[input]` parser, `ReActAgent`, plus `AnthropicLLM` (real API) and `MockLLM` (scripted, offline) backends. |
| `tools.py` | Two tools: `Calculator` (safe AST-based arithmetic, no `eval`) and `Search` (a tiny local knowledge-base lookup standing in for a real search API). |
| `test_agent.py` | Runs the loop against `MockLLM` — proves the parsing/loop/tool-dispatch mechanics work with **zero API calls**. Run this first. |
| `run_live.py` | CLI to run the same agent against the real Claude API. |

## Quickstart

```bash
# 1. See the mechanics work, offline, no API key needed:
python3 test_agent.py

# 2. Run it for real:
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python3 run_live.py "How many years passed between Cézanne's birth and Van Gogh's time in Arles?"
```

## Memory modes and token usage

The agent supports three context strategies for controlling prompt size and cost. Use the single canonical option `memory_mode`:

- `full`: keeps the original behavior with the entire transcript in the prompt.
- `compact`: keeps a short memory plus the most recent turns.
- `summary`: keeps a short memory plus a brief history summary and the most recent turns.

This is configured through `ReActAgent(..., memory_mode="summary")`.

The default is `summary`, which is the best balance for most small tool-using agents.

`memory_mode` is the only supported configuration knob for this behavior; the old compact-memory flag was removed to keep the API explicit and unambiguous.

```python
agent = ReActAgent(
    llm=llm,
    tools=TOOLS,
    tool_descriptions=TOOL_DESCRIPTIONS,
    memory_mode="summary",
)
```

The point is to preserve just enough context for the model to reason correctly while avoiding the unbounded token growth that happens when every tool result is appended forever to the prompt.

## How the loop works

Every step, the running transcript looks like:

```
Thought: <model reasons about what it knows / needs>
Action: ToolName[tool input]
Observation: <we append this after running the tool>
```

repeated until the model emits:

```
Thought: I now know the final answer
Action: Finish[the answer]
```

The whole "agent" is just:

1. Send the transcript so far to the LLM.
2. Find the first `Action: Name[` and scan to its matching `]` (depth-
   aware, so an input containing brackets survives), plus the preceding
   `Thought:`.
3. If `Finish` → return the answer. If a real tool → call it, append
   `Observation: <result>` to the transcript. If unknown → tell the
   model so it can recover. If nothing parsed → re-prompt with the
   required format, up to `max_parse_retries` times in a row.
4. Go to 1, up to `max_steps`.

That's the entire pattern — the "reasoning" is just the model's free
text; the "acting" is a string match plus a Python function call.

The memory mode does not change the core ReAct loop. It only changes what
information gets sent back to the model on the next turn, so you can trade
off between richer context and lower token cost.

## Things worth experimenting with next

- **Swap `Search` for a real API** (web search, Wikipedia, a vector
  DB) — the loop code doesn't change at all, only `tools.py` does.
- **Break the format on purpose** — feed the model a vague or
  contradictory question and watch it either recover or hit
  `hit_step_cap`. This is where you'll feel the difference between
  ReAct and more robust patterns (e.g. adding self-correction, or a
  planner that pre-commits to a multi-step plan — "Plan-and-Execute").
- **Log token usage per step** (via `response.usage` in
  `AnthropicLLM.__call__`) to see how cost scales with step count —
  relevant once you're chaining agents for a real travel/tourism use
  case with many tool calls per query.
- **Tighten the retry budget**: unparseable output now triggers a
  re-prompt (`max_parse_retries`, default 2), but a retry burns a step
  from the same `max_steps` budget as real tool calls. Whether a
  formatting stumble should cost as much as a tool call is a judgement
  call worth playing with.
- **Tune memory mode**: if you want maximum determinism, use `full`; if
  you want lower cost, prefer `compact` or `summary`.
