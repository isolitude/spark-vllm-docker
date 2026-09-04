#!/usr/bin/env python3
"""
Offline streaming smoke test for mods/fix-deepseek-v4-dsml-recovery.

Run inside a container with the mod already applied (this mod patches the
real vllm.parser tree in place, so no PYTHONPATH shim is needed once
run.sh has run):

    python3 mods/fix-deepseek-v4-dsml-recovery/patch_vllm.py
    python3 mods/fix-deepseek-v4-dsml-recovery/test_dsml_recovery.py

Or against a scratch copy:

    python3 mods/fix-deepseek-v4-dsml-recovery/patch_vllm.py verify /tmp/scratch
    PYTHONPATH=/tmp/scratch python3 mods/fix-deepseek-v4-dsml-recovery/test_dsml_recovery.py

Pure validation harness -- no network, no container mutation beyond what
patch_vllm.py already did.
"""

from __future__ import annotations

import json
import sys

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from openai.types.responses import FunctionTool

_DSML = "｜DSML｜"
TOOL_START = f"<{_DSML}tool_calls>"
TOOL_END = f"</{_DSML}tool_calls>"
INVOKE_PREFIX = f'<{_DSML}invoke name="'
INVOKE_END = f"</{_DSML}invoke>"
PARAM_CLOSE = f"</{_DSML}parameter>"

ASCII_INVOKE_PREFIX = '<invoke name="'
ASCII_INVOKE_END = "</invoke>"
ASCII_PARAM_CLOSE = "</parameter>"


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


def edit_tool() -> FunctionTool:
    return FunctionTool(
        type="function",
        name="Edit",
        description="Edit a file.",
        parameters={
            "type": "object",
            "properties": {
                "target_command": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["target_command"],
        },
    )


def make_engine(tools):
    return DeepSeekV4Parser(
        StubTokenizer(),
        tools,
        chat_template_kwargs={"enable_thinking": False},
    )


def run_turn(engine, texts, finished=True, tools=None, tool_choice="auto"):
    req = StubRequest(tools or [], tool_choice)
    tool_calls = []
    contents = []
    for i, text in enumerate(texts):
        is_fin = finished and i == len(texts) - 1
        if i == 0:
            engine._check_skip_tool_parsing(req)
        delta = engine.parse_delta(text, [], req, finished=is_fin)
        if delta:
            if delta.tool_calls:
                tool_calls.append(delta.tool_calls)
            if delta.content:
                contents.append(delta.content)
    if finished:
        # Stock engine behavior: text emitted after a tool call is deferred
        # and only flushed by a subsequent non-tool delta or the final
        # finish_streaming() call -- mirrors what the real streaming loop
        # does once generation ends.
        tail = engine.finish_streaming()
        if tail and tail.content:
            contents.append(tail.content)
    return tool_calls, contents


def all_calls(tool_calls):
    out = []
    for deltas in tool_calls:
        for tc in deltas:
            if tc.function:
                out.append((tc.function.name, tc.function.arguments))
    return out


def all_bash_args(tool_calls) -> str:
    return "|".join(
        args or "" for name, args in all_calls(tool_calls) if name == "bash"
    )


# ---------------------------------------------------------------------------
# Case 1: fullwidth invoke, missing outer wrapper (existing full-mod case)
# ---------------------------------------------------------------------------
def case_1_fullwidth_orphan_invoke():
    engine = make_engine([bash_tool()])
    text = (
        f'{INVOKE_PREFIX}bash">'
        f'echo hi'
        f'{INVOKE_END}'
    )
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    calls = all_calls(tc)
    assert any(name == "bash" for name, _ in calls), f"case1: no bash recovered: {calls!r}"
    print("  [pass] 1  fullwidth orphan invoke (no wrapper) -> recovered as real call")


# ---------------------------------------------------------------------------
# Case 2: ASCII dialect invoke, missing outer wrapper (the dominant real
# failure mode from the corpus: Claude-Code-style <invoke>/<parameter> tags)
# ---------------------------------------------------------------------------
def case_2_ascii_orphan_invoke():
    engine = make_engine([bash_tool()])
    text = (
        f'{ASCII_INVOKE_PREFIX}bash">'
        f'<parameter name="command">docker images</parameter>'
        f'{ASCII_INVOKE_END}'
    )
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    calls = all_calls(tc)
    assert any(name == "bash" for name, _ in calls), (
        f"case2: no bash recovered from ASCII dialect: {calls!r}"
    )
    print("  [pass] 2  ASCII-dialect orphan invoke (no wrapper) -> recovered as real call")


# ---------------------------------------------------------------------------
# Case 3: invalid/undeclared tool name inside an orphan invoke -> rollback
# to plain text, no crash, no spurious call.
# ---------------------------------------------------------------------------
def case_3_invalid_name_rolls_back():
    engine = make_engine([bash_tool()])
    text = f'{ASCII_INVOKE_PREFIX}NotARealTool">{ASCII_INVOKE_END}'
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    calls = all_calls(tc)
    assert not calls, f"case3: undeclared tool name should not commit a call: {calls!r}"
    raw = "".join(contents)
    assert "NotARealTool" in raw, (
        f"case3: rolled-back text should reappear as content: {raw!r}"
    )
    print("  [pass] 3  invalid tool name in orphan invoke -> rolled back to content")


# ---------------------------------------------------------------------------
# Case 4: prose that merely quotes an invoke-looking string with no
# declared tool of that name -> must not hang or falsely commit.
# ---------------------------------------------------------------------------
def case_4_prose_quoting_invoke_syntax():
    engine = make_engine([bash_tool()])
    text = (
        "For example, DSML calls look like "
        f'{ASCII_INVOKE_PREFIX}Example">{ASCII_INVOKE_END} in the docs.'
    )
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    calls = all_calls(tc)
    assert not calls, f"case4: prose should not commit a fake call: {calls!r}"
    print("  [pass] 4  prose quoting invoke syntax -> no false commit")


# ---------------------------------------------------------------------------
# Case 5: orphan closing tail (no opener at all) -> synthetic bash re-issue,
# residual absorbed, exactly one call.
# ---------------------------------------------------------------------------
def case_5_orphan_closing_tail_regen():
    engine = make_engine([bash_tool()])
    tc, contents = run_turn(
        engine,
        [PARAM_CLOSE + "\n" + INVOKE_END + "\n" + TOOL_END],
        tools=[bash_tool()],
    )
    joined = all_bash_args(tc)
    assert "re-issued" in joined, f"case5: expected regen prompt, got {joined!r}"
    parts = joined.split("|")
    assert len(parts) == 1, f"case5: expected exactly one synthetic call: {joined!r}"
    cmd = json.loads(parts[0])["command"]
    assert "DSML" not in cmd, f"case5: regen prompt must never mention DSML: {cmd!r}"
    raw = "".join(contents)
    assert PARAM_CLOSE not in raw and INVOKE_END not in raw and TOOL_END not in raw, (
        f"case5: residual leaked into content: {raw!r}"
    )
    print("  [pass] 5  orphan closing tail (no opener) -> bash re-issue, residual absorbed")


# ---------------------------------------------------------------------------
# Case 6: ASCII orphan closing tail variant.
# ---------------------------------------------------------------------------
def case_6_ascii_orphan_closing_tail_regen():
    engine = make_engine([bash_tool()])
    tc, contents = run_turn(
        engine,
        [ASCII_PARAM_CLOSE + "\n" + ASCII_INVOKE_END],
        tools=[bash_tool()],
    )
    joined = all_bash_args(tc)
    assert "re-issued" in joined, f"case6: expected regen prompt, got {joined!r}"
    raw = "".join(contents)
    assert ASCII_PARAM_CLOSE not in raw and ASCII_INVOKE_END not in raw, (
        f"case6: residual leaked into content: {raw!r}"
    )
    print("  [pass] 6  ASCII orphan closing tail -> bash re-issue, residual absorbed")


# ---------------------------------------------------------------------------
# Case 7: recovered invoke followed by its own leftover outer closer must
# not double-fire a regenerate (the outer-closer-pending guard).
# ---------------------------------------------------------------------------
def case_7_recovered_call_then_leftover_closer_no_double_regen():
    engine = make_engine([bash_tool()])
    text = (
        f'{ASCII_INVOKE_PREFIX}bash">'
        f'<parameter name="command">ls</parameter>'
        f'{ASCII_INVOKE_END}'
        f'{TOOL_END}'  # leftover outer closer with no matching opener
        "\nAll done."
    )
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    calls = all_calls(tc)
    real_calls = [(n, a) for n, a in calls if a is None or "re-issued" not in (a or "")]
    assert any(name == "bash" for name, _ in real_calls), f"case7: bash not recovered: {calls!r}"
    assert not any("re-issued" in (a or "") for _, a in calls), (
        f"case7: leftover closer must not trigger a spurious regenerate: {calls!r}"
    )
    raw = "".join(contents)
    assert "All done." in raw, f"case7: trailing content lost: {raw!r}"
    print("  [pass] 7  recovered call + leftover outer closer -> no double regen, trailing text kept")


# ---------------------------------------------------------------------------
# Case 8: undeclared bash -> orphan tail residual absorbed, no synthetic call.
# ---------------------------------------------------------------------------
def case_8_no_bash_declared():
    engine = make_engine([edit_tool()])
    tc, _ = run_turn(
        engine,
        [PARAM_CLOSE + "\n" + INVOKE_END + "\n" + TOOL_END],
        tools=[edit_tool()],
    )
    for name, _ in all_calls(tc):
        assert name != "bash", "case8: undeclared bash emitted a synthetic call"
    print("  [pass] 8  undeclared bash -> residual absorbed, no synthetic call")


# ---------------------------------------------------------------------------
# Case 9: bare parameter fragment, no invoke opener anywhere -> unrecoverable,
# should not crash, and (with bash declared) should still regenerate once
# the closing tail terminals are reached.
# ---------------------------------------------------------------------------
def case_9_bare_parameter_fragment():
    engine = make_engine([bash_tool()])
    text = (
        f'{_DSML_PARAM_OPEN}description" string="true">List all tasks{PARAM_CLOSE}\n'
        f'{INVOKE_END}\n{TOOL_END}'
    )
    tc, contents = run_turn(engine, [text], tools=[bash_tool()])
    joined = all_bash_args(tc)
    assert "re-issued" in joined, f"case9: expected regen prompt, got {joined!r}"
    print("  [pass] 9  bare parameter fragment (no invoke opener) -> bash re-issue")


_DSML_PARAM_OPEN = f"<{_DSML}parameter name=\""


_ALL = [
    ("fullwidth orphan invoke -> recovered", case_1_fullwidth_orphan_invoke),
    ("ASCII-dialect orphan invoke -> recovered", case_2_ascii_orphan_invoke),
    ("invalid tool name -> rollback to content", case_3_invalid_name_rolls_back),
    ("prose quoting invoke syntax -> no false commit", case_4_prose_quoting_invoke_syntax),
    ("orphan closing tail -> bash regenerate", case_5_orphan_closing_tail_regen),
    ("ASCII orphan closing tail -> bash regenerate", case_6_ascii_orphan_closing_tail_regen),
    ("recovered call + leftover closer -> no double regen", case_7_recovered_call_then_leftover_closer_no_double_regen),
    ("undeclared bash -> absorbed only", case_8_no_bash_declared),
    ("bare parameter fragment -> bash regenerate", case_9_bare_parameter_fragment),
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
        print(f"test_dsml_recovery: {failures}/{len(_ALL)} cases FAILED")
        sys.exit(1)
    print(f"test_dsml_recovery: all {len(_ALL)} cases passed")


if __name__ == "__main__":
    main()
