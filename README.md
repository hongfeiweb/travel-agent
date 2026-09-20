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
2. Regex out `Action: (\w+)\[(.*)\]` and the preceding `Thought:`.
3. If `Finish` → return the answer. If a real tool → call it, append
   `Observation: <result>` to the transcript. If unknown → tell the
   model so it can recover.
4. Go to 1, up to `max_steps`.

That's the entire pattern — the "reasoning" is just the model's free
text; the "acting" is a string match plus a Python function call.

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
- **Add a "no valid action" recovery step**: right now, if the model's
  output doesn't parse, the agent just stops. A more robust version
  would re-prompt with "Your last response didn't match the required
  format, please retry" instead of giving up.
