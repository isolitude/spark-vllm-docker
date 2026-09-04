#!/usr/bin/env python3
"""
Offline streaming smoke test for mods/fix-deepseek-v4-orphan-tail.

Run against a copy of the vLLM parser tree patched by this mod. For example:

    python3 mods/fix-deepseek-v4-orphan-tail/patch_vllm.py verify /tmp/vllm-raw
    PYTHONPATH=/tmp/vllm-raw \
      python3 mods/fix-deepseek-v4-orphan-tail/test_tail.py

Because the patched copy re-exposes ``vllm.parser``, set ``PYTHONPATH`` so the
patched parser wins while the real (container) ``vllm`` deps resolve. Requires
running where ``openai.types.responses.FunctionTool`` etc. are importable (the
vLLM image). Pure validation harness -- no network, no container mutation.
"""

from __future__ import annotations

import sys

# Allow an explicit patched copy to be injected; otherwise fall back to PYTHONPATH.
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

from vllm.parser.deepseek_v4 import deepseek_v4_config
from vllm.parser.engine.parser_engine import ParserEngine
from openai.types.responses import FunctionTool

_DSML = "｜DSML｜"
TOOL_START = f"<{_DSML}tool_calls>"
TOOL_END = f"</{_DSML}tool_calls>"
INVOKE_END = f"</{_DSML}invoke>"
PARAM_CLOSE = f"</{_DSML}parameter>"


class StubTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    @property
    def all_special_tokens(self) -> list[str]:
        raise AttributeError("no special tokens")

    @property
    def all_special_ids(self) -> list[int]:
        raise AttributeError("no special ids")


class StubRequest:
    def __init__(self, tools, tool_choice="auto"):
        self.tools = tools
        self.tool_choice = tool_choice
        self.include_reasoning = True


def bash_tool() -> FunctionTool:
    return FunctionTool(
        type="function",
        name="bash",
        description="Run a shell command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )


def make_engine(with_bash: bool = True):
    tools = [bash_tool()] if with_bash else []
    return (
        ParserEngine(
            StubTokenizer(),
            tools,
            parser_engine_config=deepseek_v4_config(thinking=False),
        ),
        tools,
    )


def run_turn(engine, texts, finished=True, tools=None, tool_choice="auto"):
    """Feed a list of (text, finished) chunks through ParserEngine.parse_delta.

    Returns the concatenated tool_call deltas and the full-delta content.
    """
    req = StubRequest(tools if tools is not None else engine.original_tools, tool_choice)
    tool_calls = []
    contents = []
    for i, text in enumerate(texts):
        is_fin = finished and i == len(texts) - 1
        # Re-derive tools: production sets them via request.tools in _check_skip_tool_parsing
        if i == 0:
            engine._check_skip_tool_parsing(req)
        delta = engine.parse_delta(text, [], req, finished=is_fin)
        if delta:
            if delta.tool_calls:
                tool_calls.append(delta.tool_calls)
            if delta.content:
                contents.append(delta.content)
    return tool_calls, contents


def all_bsh_args(tool_calls) -> str:
    out = []
    for deltas in tool_calls:
        for tc in deltas:
            if tc.function and tc.function.name == "bash":
                out.append(tc.function.arguments or "")
    return "|".join(out)


def case_a_regen_on_orphan_tail():
    """A lone orphan closing tail (no opener, as after a successful call) -> bash re-issue prompt."""
    engine, tools = make_engine(with_bash=True)
    engine.original_tools = tools

    # Feed only the orphan closing tail: PARAM_CLOSE absorbed, then INVOKE_END/
    # TOOL_END trigger the synthetic bash re-issue prompt. This is the exact
    # failure from issue #51914 comment #5354570612 (a turn of only closing markers).
    tc, contents = run_turn(
        engine,
        [PARAM_CLOSE + "\n" + INVOKE_END + "\n" + TOOL_END],
        finished=True,
    )
    joined = all_bsh_args(tc)
    assert "re-issued" in joined, (
        f"case A: expected bash regen prompt, got args={joined!r}"
    )
    # Double-emit regression guard: the synthetic args must be exactly ONE clean
    # JSON object -- never a trailing {} appended by the arg converter flush.
    import json as _json
    _args_list = joined.split("|")
    assert len(_args_list) == 1, (
        f"case A: expected exactly one synthetic call, got {len(_args_list)}: {joined!r}"
    )
    try:
        _cmd = _json.loads(_args_list[0]).get("command", "")
    except _json.JSONDecodeError as _e:
        raise AssertionError(
            f"case A: synthetic args are not clean JSON (trailing {{}} leak?): {joined!r}"
        ) from _e
    assert "retry" in _cmd and _cmd.endswith("tool call now."), (
        f"case A: synthetic command mismatch: {_cmd!r}"
    )
    # The residual closing markers must not leak into content unabsorbed.
    raw = "".join(contents)
    assert PARAM_CLOSE not in raw and INVOKE_END not in raw and TOOL_END not in raw, (
        f"case A: residual DSML leaked into content: {raw!r}"
    )
    print("  [pass] A  orphan closing tail -> bash re-issue prompt, residual absorbed")


def case_b_prose_quoting_dsml_no_regen():
    """Prose that merely *contains* the closers in CONTENT -> only absorbed, no regen."""
    engine, tools = make_engine(with_bash=True)
    engine.original_tools = tools
    tc, _ = run_turn(
        engine,
        [f"See {PARAM_CLOSE} here", f" and {INVOKE_END} and {TOOL_END}."],
        finished=True,
    )
    # No INVOKE_END/TOOL_END regen because those appear mid-prose (not a CONTENT
    # stray closer after a tool tail); a lone PARAM_CLOSE is absorbed silently.
    assert not tc or not all_bsh_args(tc), (
        f"case B: prose quoting closers unexpectedly triggered regen: {tc!r}"
    )
    print("  [pass] B  prose quoting DSML closers -> no regen")


def case_c_no_bash_declared():
    """bash not declared -> residual absorbed, no synthetic call."""
    engine, tools = make_engine(with_bash=False)
    engine.original_tools = tools
    tc, _ = run_turn(
        engine,
        [PARAM_CLOSE + "\n" + INVOKE_END + "\n" + TOOL_END],
        finished=True,
    )
    for deltas in tc:
        for c in deltas:
            assert not (c.function and c.function.name == "bash"), (
                "case C: un-declared bash emitted a call"
            )
    print("  [pass] C  undeclared bash -> residual absorbed, no synthetic call")


def case_d_no_tools_request():
    """tools=[] + tool_choice=none -> no crash, no synthetic call."""
    engine, tools = make_engine(with_bash=True)
    engine.original_tools = tools
    tc, _ = run_turn(
        engine,
        [PARAM_CLOSE + "\n" + INVOKE_END + "\n" + TOOL_END],
        finished=True,
        tools=[],
        tool_choice="none",
    )
    assert tc == [], f"case D: tool_choice=none produced tool deltas: {tc!r}"
    print("  [pass] D  no-tools request -> no crash, no synthetic call")


_ALL = [
    ("orphan closing tail -> bash regenerate, residual absorbed", case_a_regen_on_orphan_tail),
    ("prose quoting DSML -> no regen", case_b_prose_quoting_dsml_no_regen),
    ("undeclared bash -> absorbed only", case_c_no_bash_declared),
    ("no-tools request -> no crash/call", case_d_no_tools_request),
]


def main() -> None:
    failures = 0
    for name, fn in _ALL:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"test_tail: {failures}/{len(_ALL)} cases FAILED")
        sys.exit(1)
    print(f"test_tail: all {len(_ALL)} cases passed")


if __name__ == "__main__":
    main()
