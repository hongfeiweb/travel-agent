"""
Tools available to the ReAct agent.

Each tool is just a Python callable: str -> str.
Keep tools small, deterministic, and easy to reason about — the point
of this sandbox is to see the agent's Thought/Action/Observation loop
clearly, not to build production-grade tools.
"""

import ast
import operator as op

# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------
# A safe arithmetic evaluator (no eval()) supporting + - * / ** // % and
# parentheses. This is the classic "give the LLM a calculator" ReAct demo.

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


# ---------------------------------------------------------------------------
# Tool 2: Mock knowledge base ("search")
# ---------------------------------------------------------------------------
# A tiny local lookup table standing in for a real search/retrieval API.
# This keeps the sandbox fully offline and deterministic. Swap this out
# for a real web-search or vector-DB tool once you're comfortable with
# the loop mechanics.

_KB = {
    "eiffel tower height": "The Eiffel Tower is 330 meters tall (including antennas).",
    "tallest mountain": "Mount Everest is the tallest mountain above sea level, at 8,849 meters.",
    "capital of provence": "Aix-en-Provence is the historic capital of Provence; Marseille is the largest city in the region.",
    "cezanne birthplace": "Paul Cézanne was born in Aix-en-Provence, France, in 1839.",
    "van gogh arles": "Vincent van Gogh lived in Arles, in the south of France, from 1888 to 1889, producing over 300 works there.",
    "population of tokyo": "The Tokyo metropolitan area has a population of roughly 37 million people.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
}


def search(query: str) -> str:
    """Look up a fact in a small local knowledge base (mock search tool)."""
    q = query.strip().lower()
    # exact match first
    if q in _KB:
        return _KB[q]
    # fuzzy: return any entry whose key shares words with the query
    q_words = set(q.split())
    best_key, best_overlap = None, 0
    for key in _KB:
        overlap = len(q_words & set(key.split()))
        if overlap > best_overlap:
            best_key, best_overlap = key, overlap
    if best_key and best_overlap > 0:
        return _KB[best_key]
    return f"No results found for '{query}'. Try rephrasing, or answer from general knowledge."


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS = {
    "Calculator": calculator,
    "Search": search,
}

TOOL_DESCRIPTIONS = {
    "Calculator": "Evaluate an arithmetic expression. Input: a math expression, e.g. '12*7'.",
    "Search": "Look up a short factual snippet from a small local knowledge base. Input: a plain-text query.",
}
