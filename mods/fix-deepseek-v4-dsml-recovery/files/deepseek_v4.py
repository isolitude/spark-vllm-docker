# __spark_dsml_recovery__
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Spark recovery extension
-------------------------
DeepSeek-V4-Flash-0731 intermittently drops or corrupts this format under
long context. Sampled failures fall into three real, distinct shapes (see
mods/fix-deepseek-v4-dsml-recovery/README.md for the corpus evidence):

1. **Dialect confusion**: a complete ``<invoke name="X">...</invoke>`` block
   using the *ASCII* tag dialect (the same one Claude Code's own tool-call
   protocol uses) instead of fullwidth DSML tags, with the outer
   ``<｜DSML｜tool_calls>`` wrapper missing. Also covers a genuine fullwidth
   ``<｜DSML｜invoke ...>`` block missing its outer wrapper. This is the
   majority failure shape and is fully recoverable: the tool name and
   arguments are intact, only the wrapper is gone.
2. **Bare parameter fragments**: ``<｜DSML｜parameter>``/``<parameter>``
   elements with no invoke opener at all. Unrecoverable — there's no tool
   name to recover.
3. **Orphan closing tail**: stray closers (``</invoke>``, ``</tool_calls>``,
   ``</parameter>``) landing in CONTENT after a call has already resolved,
   with no real content. Unrecoverable.

For (1): buffer the candidate invoke provisionally (regardless of which
dialect opened it), validate the completed tool name against the tools
declared on the current request, and commit only once the tool name is
confirmed valid — otherwise roll the buffered text back to ordinary content.
This mirrors the "provisional hold" design proven in the vLLM upstream fix
for vllm-project/vllm#51914 (PR #52645), extended to also recognize the
ASCII invoke dialect.

For (2) and (3): StreamingParserEngine absorbs the stray terminal and emits
a synthetic ``bash`` tool call asking the model to re-issue its last tool
call (never mentions "DSML" in that prompt — doing so was observed to
increase the frequency of this exact failure mode).
"""

from __future__ import annotations

import contextlib
import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

# ASCII dialect: same shape, no fullwidth glyphs. DSML_INVOKE_NAME_END
# ('">') is byte-identical in both dialects so it needs no ASCII twin.
ASCII_INVOKE_PREFIX = '<invoke name="'
ASCII_INVOKE_END = "</invoke>"
ASCII_PARAM_CLOSE = "</parameter>"

_ESCAPED_DSML = re.escape(_DSML)
# The fullwidth DSML dialect always carries string="true|false"; the ASCII
# dialect (the same one Claude Code's own tool-call protocol uses) commonly
# omits it entirely -- every ASCII <parameter> value is then just a string.
# ``string`` group is None when the attribute is absent.
_PARAM_RE = re.compile(
    rf'<(?:{_ESCAPED_DSML})?parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?>'
    rf"(.*?)</(?:{_ESCAPED_DSML})?parameter>",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<(?:{_ESCAPED_DSML})?parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?>'
    rf"(.*)$",
    re.DOTALL,
)


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    last_end = 0
    for m in _PARAM_RE.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str in ("true", None):
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        pm = _PARTIAL_PARAM_RE.search(raw_args, last_end)
        if pm:
            name, is_str, value = pm.group(1), pm.group(2), pm.group(3)
            if is_str in ("true", None):
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
            "ASCII_INVOKE_PREFIX": ASCII_INVOKE_PREFIX,
            "ASCII_INVOKE_END": ASCII_INVOKE_END,
            "ASCII_PARAM_CLOSE": ASCII_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "ASCII_INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
                commit_provisional_tool_call=True,
            ),
            (ParserState.TOOL_ARGS, "ASCII_INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
                commit_provisional_tool_call=True,
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "ASCII_INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # --- Spark DSML recovery: provisional invoke without a wrapper ---
            # DeepSeek V4 can intermittently omit or corrupt the outer
            # <｜DSML｜tool_calls> wrapper, or emit the invoke in the ASCII
            # dialect. Hold the candidate provisional until the function name
            # is complete and the request actually declares it.
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.CONTENT, "ASCII_INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                provisional_tool_call=True,
            ),
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                provisional_tool_call=True,
            ),
            (ParserState.REASONING, "ASCII_INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                provisional_tool_call=True,
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args
        self._recovery_request_tools: list[Tool] = list(tools or [])
        self._recovery_suppressed = False
        if hasattr(self._engine, "recovery_tool_name_validator"):
            self._engine.recovery_tool_name_validator = self._can_recover_tool_name

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)

    def _check_skip_tool_parsing(self, request) -> None:
        super()._check_skip_tool_parsing(request)
        self._recovery_request_tools = list(getattr(request, "tools", None) or [])
        self._recovery_suppressed = getattr(request, "tool_choice", None) == "none"

    def _can_recover_tool_name(self, name: str) -> bool:
        return bool(
            name
            and self._recovery_request_tools
            and not self._recovery_suppressed
            and find_tool_name(self._recovery_request_tools, name)
        )
