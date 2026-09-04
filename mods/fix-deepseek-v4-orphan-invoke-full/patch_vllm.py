#!/usr/bin/env python3
"""
Inline patch for vLLM DeepSeek-V4: full orphan-invoke recovery, ported from
upstream vLLM PR #52645 (Fixes #51914).

Why this exists
---------------
DeepSeek-V4-Flash-0731 intermittently omits or corrupts the outer
<｜DSML｜tool_calls> start wrapper (observed variant spells it ``toolcalls``)
while still emitting a complete <｜DSML｜invoke ...>...</｜DSML｜invoke> block.
The parser state machine then falls through and returns the whole DSML block
as plain text, so the structured tool call is lost.

PR #52645 added a provisional-recovery mechanism: buffer the candidate invoke,
validate the completed function name against the tools declared by the current
request, commit only on the invoke-end transition, and otherwise restore the
text as ordinary content. It also keeps V3.2 ``function_calls`` wrappers
verbatim so their inner invokes don't enter the V4 recovery path.

PR #52645 targets vLLM *current-main*. The running image is a fork
(``vllm/parser/engine/streaming_parser_engine.py`` is a simpler variant that
does not thread ``token_count``). This patch re-implements the PR's logic onto
that fork so it can be applied to ``/usr/local/lib/python3.12/dist-packages``.

Idempotent: safe to re-run. Coexists with the lighter
``mods/fix-deepseek-v4-orphan-invoke``: this patch first strips the two plain
(non-provisional) ``INVOKE_PREFIX`` transitions that mod injects, then installs
the provisional variants, so the two mods do not fight.
"""

from __future__ import annotations

import glob
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SITE = "/usr/local/lib/python3.12/dist-packages"
VLLM_PARSER = f"{_SITE}/vllm/parser"
VLLM_TOOL = f"{_SITE}/vllm/tool_parsers"

MODE = "verify"
if len(sys.argv) > 1 and sys.argv[1] == "verify":
    # Called with a target prefix under which a copy of vllm/ sits.
    if len(sys.argv) > 2:
        _SITE = sys.argv[2]
        VLLM_PARSER = f"{_SITE}/vllm/parser"
        VLLM_TOOL = f"{_SITE}/vllm/tool_parsers"


def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# Generic: replace an exact anchor once, error if missing. Markers give
# idempotency (a marker already present => skip that whole file).
def _require_replace(
    content: str, anchor: str, replacement: str, what: str
) -> str:
    if anchor not in content:
        _abort(f"could not find anchor for {what}")
    if content.count(anchor) != 1:
        _abort(f"anchor for {what} is not unique ({content.count(anchor)} hits)")
    return content.replace(anchor, replacement)


# ---------------------------------------------------------------------------
# 1. vllm/parser/engine/parser_engine_config.py
# ---------------------------------------------------------------------------
_FULL = "__spark_orphan_full__"


def patch_engine_config(path: str) -> None:
    content = _read(path)
    if _FULL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # ParserState: add FOREIGN_* states before TOOL_PREAMBLE (any position is
    # fine, they must just exist).
    anchor_state = (
        "    TOOL_PREAMBLE = auto()\n"
    )
    repl_state = (
        "    TOOL_PREAMBLE = auto()\n"
        "    FOREIGN_BLOCK = auto()\n"
        "    FOREIGN_REASONING_BLOCK = auto()\n"
    )
    content = _require_replace(
        content, anchor_state, repl_state, "ParserState FOREIGN states"
    )

    # Transition: add the two recovery flags.
    anchor_tr = (
        "@dataclass(frozen=True, slots=True)\n"
        "class Transition:\n"
        "    next_state: ParserState\n"
        "    events: tuple[EventType, ...] = field(default_factory=tuple)\n"
        "    skip_in_token_id_mode: bool = False\n"
    )
    repl_tr = (
        "@dataclass(frozen=True, slots=True)\n"
        "class Transition:\n"
        "    next_state: ParserState\n"
        "    events: tuple[EventType, ...] = field(default_factory=tuple)\n"
        "    skip_in_token_id_mode: bool = False\n"
        "    # Mark this transition as a provisional tool-call recovery path:\n"
        "    # the engine buffers its events, validates the completed tool\n"
        "    # name through the parser's validator, and commits only at the\n"
        "    # invoke-end transition.\n"
        "    provisional_tool_call: bool = False\n"
        "    # Commit a buffered provisional call at this transition; recovery\n"
        "    # paths that leave tool arguments any other way are restored as\n"
        "    # ordinary text.\n"
        "    commit_provisional_tool_call: bool = False\n"
        f"    {_FULL} = False\n"
    )
    content = _require_replace(content, anchor_tr, repl_tr, "Transition flags")

    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
# 2. vllm/parser/deepseek_v4.py
# ---------------------------------------------------------------------------
_PATCHED_DSV4 = "__spark_orphan_full_deepseek_v4__"
_NEW_RECOVERY_MARK = "__spark_orphan_full_recovery__"


def patch_deepseek_v4(path: str) -> None:
    content = _read(path)
    if _PATCHED_DSV4 in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # 2a. Strip the older lightweight mod's plain INVOKE_PREFIX transitions so
    # they don't collide with the provisional variants we install below.
    OLD_MARK = "# orphan INVOKE_PREFIX fallback"
    if OLD_MARK in content:
        old_fallback = (
            OLD_MARK
            + ": recover a <\N{FULLWIDTH VERTICAL LINE}DSML\N{FULLWIDTH VERTICAL LINE}invoke ...> block "
            "emitted without an opening <\N{FULLWIDTH VERTICAL LINE}DSML\N{FULLWIDTH VERTICAL LINE}tool_calls> wrapper\n"
            "            (ParserState.CONTENT, \"INVOKE_PREFIX\"): Transition(\n"
            "                ParserState.TOOL_NAME,\n"
            "                (EventType.TOOL_CALL_START,),\n"
            "            ),\n"
            "            (ParserState.REASONING, \"INVOKE_PREFIX\"): Transition(\n"
            "                ParserState.TOOL_NAME,\n"
            "                (EventType.REASONING_END, EventType.TOOL_CALL_START,),\n"
            "            ),\n"
        )
        if old_fallback in content:
            content = content.replace(old_fallback, "")
            print("  [clean] removed previous fallback INVOKE_PREFIX transitions")
        else:
            _abort("previous fallback marker present but block shape changed")
    else:
        print("  [clean] no previous fallback transitions found")

    # 2b. imports: add find_tool_name
    anchor_import = "from vllm.tool_parsers.utils import find_tool_properties\n"
    if "find_tool_name" not in content:
        content = _require_replace(
            content,
            anchor_import,
            "from vllm.tool_parsers.utils import find_tool_name, find_tool_properties\n",
            "import find_tool_name",
        )

    # 2c. constants
    anchor_const = 'DSML_PARAM_CLOSE = f"</{_DSML}parameter>"\n'
    repl_const = (
        'DSML_PARAM_CLOSE = f"</{_DSML}parameter>"\n'
        'DSML_FOREIGN_TOOL_START = f"<{_DSML}function_calls>"\n'
        'DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"\n'
    )
    if "DSML_FOREIGN_TOOL_START" not in content:
        content = _require_replace(
            content, anchor_const, repl_const, "FOREIGN_TOOL constants"
        )

    # 2d. terminals
    anchor_term = (
        '            "PARAM_CLOSE": DSML_PARAM_CLOSE,\n'
    )
    if "FOREIGN_START" not in content:
        repl_term = (
            '            "PARAM_CLOSE": DSML_PARAM_CLOSE,\n'
            '            "FOREIGN_START": DSML_FOREIGN_TOOL_START,\n'
            '            "FOREIGN_END": DSML_FOREIGN_TOOL_END,\n'
        )
        content = _require_replace(content, anchor_term, repl_term, "FOREIGN terminals")

    # 2e. transitions: foreign blocks + provisional INVOKE_PREFIX + commit flag
    # Insert foreign + provisional transitions just before the existing
    # TOOL_PREAMBLE->INVOKE_PREFIX entry (order is irrelevant in the dict).
    anchor_tr = (
        '            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(\n'
    )
    new_tr = (
        '# Keep V3.2 function_calls wrappers verbatim so their inner invokes\n'
        '# cannot enter the V4 orphan-recovery path.\n'
        '            (ParserState.CONTENT, "FOREIGN_START"): Transition(\n'
        '                ParserState.FOREIGN_BLOCK,\n'
        '                (EventType.TEXT_CHUNK,),\n'
        '            ),\n'
        '            (ParserState.FOREIGN_BLOCK, "FOREIGN_END"): Transition(\n'
        '                ParserState.CONTENT,\n'
        '                (EventType.TEXT_CHUNK,),\n'
        '            ),\n'
        '            (ParserState.FOREIGN_BLOCK, "TOOL_START"): Transition(\n'
        '                ParserState.TOOL_PREAMBLE,\n'
        '                (),\n'
        '            ),\n'
        '            (ParserState.REASONING, "FOREIGN_START"): Transition(\n'
        '                ParserState.FOREIGN_REASONING_BLOCK,\n'
        '                (EventType.REASONING_CHUNK,),\n'
        '            ),\n'
        '            (ParserState.FOREIGN_REASONING_BLOCK, "FOREIGN_END"): Transition(\n'
        '                ParserState.REASONING,\n'
        '                (EventType.REASONING_CHUNK,),\n'
        '            ),\n'
        '            (ParserState.FOREIGN_REASONING_BLOCK, "TOOL_START"): Transition(\n'
        '                ParserState.TOOL_PREAMBLE,\n'
        '                (EventType.REASONING_END,),\n'
        '            ),\n'
        '# DeepSeek V4 can intermittently omit or corrupt the outer\n'
        '# <\N{FULLWIDTH VERTICAL LINE}DSML\N{FULLWIDTH VERTICAL LINE}tool_calls> wrapper while still emitting a complete\n'
        '# invoke. Hold the candidate provisional until the function name is\n'
        '# complete and the request actually declared it.\n'
        '            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(\n'
        '                ParserState.TOOL_NAME,\n'
        '                (EventType.REASONING_END, EventType.TOOL_CALL_START,),\n'
        '                provisional_tool_call=True,\n'
        '            ),\n'
        '            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(\n'
        '                ParserState.TOOL_NAME,\n'
        '                (EventType.TOOL_CALL_START,),\n'
        '                provisional_tool_call=True,\n'
        '            ),\n'
        '            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(\n'
    )
    if "provisional_tool_call=True" not in content:
        content = _require_replace(content, anchor_tr, new_tr, "provisional transitions")

    # 2f. commit flag on INVOKE_END -> TOOL_BETWEEN
    anchor_commit = (
        '            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(\n'
        '                ParserState.TOOL_BETWEEN,\n'
        '                (EventType.TOOL_CALL_END,),\n'
        '            ),\n'
    )
    repl_commit = (
        '            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(\n'
        '                ParserState.TOOL_BETWEEN,\n'
        '                (EventType.TOOL_CALL_END,),\n'
        '                commit_provisional_tool_call=True,\n'
        '            ),\n'
    )
    if "commit_provisional_tool_call=True" not in content:
        content = _require_replace(content, anchor_commit, repl_commit, "commit flag")

    # 2g. content_events: add FOREIGN states
    anchor_content = (
        '            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,\n'
    )
    if "FOREIGN_BLOCK" not in content and "FOREIGN_REASONING_BLOCK" not in content:
        repl_content = (
            '            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,\n'
            '            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,\n'
            '            ParserState.FOREIGN_REASONING_BLOCK: EventType.REASONING_CHUNK,\n'
        )
        content = _require_replace(content, anchor_content, repl_content, "FOREIGN content_events")

    # 2h. class: __init__ recovery binding + validator methods
    anchor_init = (
        "        self._arg_converter = self._convert_args\n"
    )
    repl_init = (
        "        self._arg_converter = self._convert_args\n"
        "        self._recovery_request_tools: list[Tool] = list(tools or [])\n"
        "        self._recovery_suppressed = False\n"
        "        if hasattr(self._engine, \"recovery_tool_name_validator\"):\n"
        "            self._engine.recovery_tool_name_validator = self._can_recover_tool_name\n"
    )
    content = _require_replace(content, anchor_init, repl_init, "recovery __init__ binding")

    # 2h.2 add methods after _convert_args (end of class)
    anchor_methods = (
        "        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)\n"
        "        return _unwrap_wrapper_args(result, self._tools, func_name)\n"
    )
    repl_methods = (
        "        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)\n"
        "        return _unwrap_wrapper_args(result, self._tools, func_name)\n"
        "\n"
        "    def _check_skip_tool_parsing(self, request) -> None:\n"
        "        super()._check_skip_tool_parsing(request)\n"
        "        self._recovery_request_tools = list(getattr(request, \"tools\", None) or [])\n"
        "        self._recovery_suppressed = getattr(request, \"tool_choice\", None) == \"none\"\n"
        "\n"
        "    def _can_recover_tool_name(self, name: str) -> bool:\n"
        "        return bool(\n"
        "            name\n"
        "            and self._recovery_request_tools\n"
        "            and not self._recovery_suppressed\n"
        "            and find_tool_name(self._recovery_request_tools, name)\n"
        "        )\n"
    )
    if "def _can_recover_tool_name" not in content:
        content = _require_replace(content, anchor_methods, repl_methods, "validator methods")

    # 2i. module marker for idempotency
    content = content + f"\n# {_PATCHED_DSV4}\n"

    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
# 3. vllm/parser/engine/streaming_parser_engine.py
# ---------------------------------------------------------------------------
def patch_streaming_engine(path: str) -> None:
    content = _read(path)
    if _FULL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # 3a. imports: Callable
    anchor_imp = "from collections.abc import Sequence\n"
    repl_imp = "from collections.abc import Callable, Sequence\n"
    if "Callable, Sequence" not in content:
        content = _require_replace(content, anchor_imp, repl_imp, "import Callable")

    # 3b. __init__: validator attr
    anchor_attr = (
        "        self.skip_tool_parsing = False\n"
    )
    repl_attr = (
        "        self.skip_tool_parsing = False\n"
        "        # Optional parser-owned validator used only by provisional\n"
        "        # tool-call transitions. Intentionally survives reset() so the\n"
        "        # owning ParserEngine can bind a stable callback once.\n"
        "        self.recovery_tool_name_validator: Callable[[str], bool] | None = None\n"
    )
    content = _require_replace(content, anchor_attr, repl_attr, "validator attr")

    # 3c. reset(): recovery fields
    anchor_reset = (
        "        self._in_skipped_tool_span = False\n"
        "        self._reset_args_state()\n"
    )
    repl_reset = (
        "        self._in_skipped_tool_span = False\n"
        "        self._reset_args_state()\n"
        "        self._recovery_hold_active = False\n"
        "        self._recovery_hold_events: list[SemanticEvent] = []\n"
        "        self._recovery_hold_raw = \"\"\n"
        "        self._recovery_hold_name = \"\"\n"
        "        self._recovery_prior_state = self.state\n"
        "        self._recovery_prior_tool_index = -1\n"
        "        self._recovery_outer_closer_pending = False\n"
    )
    content = _require_replace(content, anchor_reset, repl_reset, "reset recovery fields")

    # 3d. finish(): abort hold + FOREIGN end states
    anchor_fin_abort = (
        "        events.extend(self._process_lex_tokens(self._lexer.flush()))\n"
        "\n"
        "        if self._args_buffer:\n"
    )
    repl_fin_abort = (
        "        events.extend(self._process_lex_tokens(self._lexer.flush()))\n"
        "\n"
        "        if self._recovery_hold_active:\n"
        "            events.extend(self._abort_recovery_hold())\n"
        "\n"
        "        if self._args_buffer:\n"
    )
    content = _require_replace(content, anchor_fin_abort, repl_fin_abort, "finish abort hold")

    anchor_fin_reasoning = (
        "        elif self.state == ParserState.REASONING:\n"
        "            events.append(\n"
        "                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)\n"
        "            )\n"
        "            self.state = ParserState.CONTENT\n"
    )
    repl_fin_reasoning = (
        "        elif self.state in (\n"
        "            ParserState.REASONING,\n"
        "            ParserState.FOREIGN_REASONING_BLOCK,\n"
        "        ):\n"
        "            events.append(\n"
        "                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)\n"
        "            )\n"
        "            self.state = ParserState.CONTENT\n"
        "        elif self.state == ParserState.FOREIGN_BLOCK:\n"
        "            self.state = ParserState.CONTENT\n"
    )
    content = _require_replace(
        content, anchor_fin_reasoning, repl_fin_reasoning, "finish FOREIGN states"
    )

    # 3e. _on_terminal: outer-closer absorb + recover-none branches
    anchor_onterm_head = (
        "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
        "        key = (self.state, terminal)\n"
        "        transition = self.config.transitions.get(key)\n"
        "\n"
        "        if transition is None:\n"
        "            if self._has_drops and terminal == DROP_TERMINAL:\n"
        "                return []\n"
        "            # The projected skip state may not define the wrapper closer.\n"
        "            if self.skip_tool_parsing and terminal in self._tool_exit_terminals:\n"
        "                self._in_skipped_tool_span = False\n"
        "            return self._emit_for_state(value)\n"
    )
    repl_onterm_head = (
        "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
        "        if (\n"
        "            self._recovery_outer_closer_pending\n"
        "            and self.state == ParserState.CONTENT\n"
        "            and terminal in self._tool_exit_terminals\n"
        "        ):\n"
        "            self._recovery_outer_closer_pending = False\n"
        "            return []\n"
        "\n"
        "        key = (self.state, terminal)\n"
        "        transition = self.config.transitions.get(key)\n"
        "\n"
        "        if transition is None:\n"
        "            # DROP_TERMINAL must never enter provisional recovery buffers.\n"
        "            if self._has_drops and terminal == DROP_TERMINAL:\n"
        "                return []\n"
        "            if self._recovery_hold_active and self.state == ParserState.TOOL_NAME:\n"
        "                events = self._abort_recovery_hold()\n"
        "                events.extend(self._on_terminal(terminal, value))\n"
        "                return events\n"
        "            if self._recovery_hold_active and self.state == ParserState.TOOL_ARGS:\n"
        "                # DSML parameter closers are terminals but deliberately\n"
        "                # have no state transition; during recovery they stay\n"
        "                # buffered as arguments.\n"
        "                return self._emit_for_state(value)\n"
        "            # The projected skip state may not define the wrapper closer.\n"
        "            if self.skip_tool_parsing and terminal in self._tool_exit_terminals:\n"
        "                self._in_skipped_tool_span = False\n"
        "            return self._emit_for_state(value)\n"
        "\n"
        "        if self.skip_tool_parsing and (\n"
        "            transition.provisional_tool_call or self._recovery_hold_active\n"
        "        ):\n"
        "            return self._apply_transition(transition, value)\n"
    )
    content = _require_replace(
        content, anchor_onterm_head, repl_onterm_head, "_on_terminal recovery head"
    )

    # 3f. _emit_for_state -> wrapper + _emit_for_state_now + hold helpers
    anchor_emit = (
        "    def _emit_for_state(self, text: str) -> list[SemanticEvent]:\n"
        "        if self.state == ParserState.MESSAGE_HEADER:\n"
        "            self._message_header_buffer += text\n"
        "            return []\n"
    )
    repl_emit = (
        "    def _emit_for_state(self, text: str) -> list[SemanticEvent]:\n"
        "        if self._recovery_hold_active:\n"
        "            return self._hold_recovery_text(text)\n"
        "        return self._emit_for_state_now(text)\n"
        "\n"
        "    def _hold_recovery_text(self, text: str) -> list[SemanticEvent]:\n"
        "        self._recovery_hold_raw += text\n"
        "\n"
        "        if self.state == ParserState.TOOL_NAME:\n"
        "            self._recovery_hold_name += text\n"
        "            # Tool names are expected to be compact. Avoid delaying an\n"
        "            # entire response if ordinary prose happens to quote an\n"
        "            # unterminated invoke marker.\n"
        "            if len(self._recovery_hold_name) > 256 or \"\\n\" in self._recovery_hold_name:\n"
        "                return self._abort_recovery_hold()\n"
        "\n"
        "        self._recovery_hold_events.extend(self._emit_for_state_now(text))\n"
        "        return []\n"
        "\n"
        "    def _emit_for_state_now(self, text: str) -> list[SemanticEvent]:\n"
        "        if self.state == ParserState.MESSAGE_HEADER:\n"
        "            self._message_header_buffer += text\n"
        "            return []\n"
    )
    content = _require_replace(
        content, anchor_emit, repl_emit, "_emit_for_state wrapper"
    )

    # 3g. _apply_transition -> recovery dispatch + _run_transition (rename body)
    # The container's _apply_transition body (without token_count). We turn the
    # existing method into _run_transition and add a thin dispatch in front.
    anchor_apply = (
        "    def _apply_transition(\n"
        "        self,\n"
        "        transition: Transition,\n"
        "        value: str,\n"
        "    ) -> list[SemanticEvent]:\n"
        "        events: list[SemanticEvent] = []\n"
    )
    repl_apply = (
        "    def _apply_transition(\n"
        "        self,\n"
        "        transition: Transition,\n"
        "        value: str,\n"
        "    ) -> list[SemanticEvent]:\n"
        "        if self._recovery_hold_active:\n"
        "            return self._advance_recovery_hold(transition, value)\n"
        "        if transition.provisional_tool_call:\n"
        "            if self.recovery_tool_name_validator is None:\n"
        "                return self._emit_for_state(value)\n"
        "            return self._begin_recovery_hold(transition, value)\n"
        "        if (\n"
        "            self._recovery_outer_closer_pending\n"
        "            and self.state == ParserState.CONTENT\n"
        "            and transition.next_state in self._TOOL_STATES\n"
        "        ):\n"
        "            self._recovery_outer_closer_pending = False\n"
        "        return self._run_transition(transition, value)\n"
        "\n"
        "    def _begin_recovery_hold(\n"
        "        self,\n"
        "        transition: Transition,\n"
        "        value: str,\n"
        "    ) -> list[SemanticEvent]:\n"
        "        prior_state = self.state\n"
        "        prior_tool_index = self.tool_index\n"
        "        held_events = self._run_transition(transition, value)\n"
        "        self._recovery_hold_active = True\n"
        "        self._recovery_hold_events = held_events\n"
        "        self._recovery_hold_raw = value\n"
        "        self._recovery_hold_name = \"\"\n"
        "        self._recovery_prior_state = prior_state\n"
        "        self._recovery_prior_tool_index = prior_tool_index\n"
        "        return []\n"
        "\n"
        "    def _advance_recovery_hold(\n"
        "        self,\n"
        "        transition: Transition,\n"
        "        value: str,\n"
        "    ) -> list[SemanticEvent]:\n"
        "        self._recovery_hold_raw += value\n"
        "\n"
        "        if self.state == ParserState.TOOL_NAME:\n"
        "            validator = self.recovery_tool_name_validator\n"
        "            if validator is None or not validator(self._recovery_hold_name):\n"
        "                return self._abort_recovery_hold()\n"
        "            self._recovery_hold_events.extend(\n"
        "                self._run_transition(transition, value)\n"
        "            )\n"
        "            return []\n"
        "\n"
        "        if self.state == ParserState.TOOL_ARGS:\n"
        "            if not transition.commit_provisional_tool_call:\n"
        "                return self._abort_recovery_hold()\n"
        "            transition_events = self._run_transition(transition, value)\n"
        "            self._recovery_hold_events.extend(transition_events)\n"
        "            events = self._recovery_hold_events\n"
        "            self._clear_recovery_hold()\n"
        "            # A recovered invoke started outside a valid outer tool\n"
        "            # wrapper. Do not leave it in TOOL_BETWEEN: subsequent text\n"
        "            # is content, and any following bare invoke re-enters the\n"
        "            # provisional path.\n"
        "            self.state = ParserState.CONTENT\n"
        "            self._recovery_outer_closer_pending = True\n"
        "            return events\n"
        "\n"
        "        return self._abort_recovery_hold()\n"
        "\n"
        "    def _abort_recovery_hold(self) -> list[SemanticEvent]:\n"
        "        raw = self._recovery_hold_raw\n"
        "        self.state = self._recovery_prior_state\n"
        "        self.tool_index = self._recovery_prior_tool_index\n"
        "        self._reset_args_state()\n"
        "        self._clear_recovery_hold()\n"
        "        return self._emit_for_state_now(raw)\n"
        "\n"
        "    def _clear_recovery_hold(self) -> None:\n"
        "        self._recovery_hold_active = False\n"
        "        self._recovery_hold_events = []\n"
        "        self._recovery_hold_raw = \"\"\n"
        "        self._recovery_hold_name = \"\"\n"
        "\n"
        "    def _run_transition(\n"
        "        self,\n"
        "        transition: Transition,\n"
        "        value: str,\n"
        "    ) -> list[SemanticEvent]:\n"
        "        events: list[SemanticEvent] = []\n"
    )
    content = _require_replace(
        content, anchor_apply, repl_apply, "_apply_transition dispatch + helpers"
    )

    # 3h. module marker
    content = content + f"\n# {_FULL}\n"
    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
# 4. vllm/parser/engine/adapters.py
# ---------------------------------------------------------------------------
def patch_adapters(path: str) -> None:
    content = _read(path)
    if _FULL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # 4a. extract_reasoning: sync tools before reasoning scan
    anchor_extract = (
        "    def extract_reasoning(\n"
        "        self,\n"
        "        model_output: str,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "    ) -> tuple[str | None, str | None]:\n"
        "        with self._skip_tool_parsing():\n"
    )
    repl_extract = (
        "    def extract_reasoning(\n"
        "        self,\n"
        "        model_output: str,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "    ) -> tuple[str | None, str | None]:\n"
        "        self.adjust_request(request)\n"
        "        with self._skip_tool_parsing():\n"
    )
    content = _require_replace(
        content, anchor_extract, repl_extract, "adapters extract_reasoning tools sync"
    )

    # 4b. adjust_request: also push tool/choice into the reasoning-side engine
    anchor_adjust = (
        "    def adjust_request(\n"
        "        self,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "    ) -> ChatCompletionRequest | ResponsesRequest:\n"
        "        return self._parser_engine.adjust_request(request)\n"
    )
    repl_adjust = (
        "    def adjust_request(\n"
        "        self,\n"
        "        request: ChatCompletionRequest | ResponsesRequest,\n"
        "    ) -> ChatCompletionRequest | ResponsesRequest:\n"
        "        request = self._parser_engine.adjust_request(request)\n"
        "        with self._skip_tool_parsing():\n"
        "            self._parser_engine._check_skip_tool_parsing(request)\n"
        "        return request\n"
    )
    content = _require_replace(
        content, anchor_adjust, repl_adjust, "adapters adjust_request tools sync"
    )

    content = content + f"\n# {_FULL}\n"
    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
# 5. vllm/parser/engine/parser_engine.py
# ---------------------------------------------------------------------------
def patch_parser_engine(path: str) -> None:
    content = _read(path)
    if _FULL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    anchor_finish = (
        "    def finish_streaming(self) -> DeltaMessage | None:\n"
        "        events = self._engine.finish()\n"
        "        if events or self._deferred_content:\n"
        "            return self._events_to_delta(events, finished=True)\n"
        "        return None\n"
    )
    repl_finish = (
        "    def finish_streaming(self) -> DeltaMessage | None:\n"
        "        events = self._engine.finish()\n"
        "        if events or self._deferred_content or self._deferred_reasoning:\n"
        "            delta = self._events_to_delta(events, finished=True)\n"
        "            return self._strip_trailing_reasoning(delta)\n"
        "        return None\n"
    )
    content = _require_replace(
        content, anchor_finish, repl_finish, "parser_engine finish_streaming"
    )

    content = content + f"\n# {_FULL}\n"
    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    files = [
        f"{VLLM_PARSER}/engine/parser_engine_config.py",
        f"{VLLM_PARSER}/deepseek_v4.py",
        f"{VLLM_PARSER}/engine/streaming_parser_engine.py",
        f"{VLLM_PARSER}/engine/adapters.py",
        f"{VLLM_PARSER}/engine/parser_engine.py",
    ]
    ok = True
    for fp in files:
        p = glob.glob(fp)
        if not p:
            print(f"  MISSING {fp}", file=sys.stderr)
            ok = False
            continue
        patch_fns = {
            "parser_engine_config.py": patch_engine_config,
            "deepseek_v4.py": patch_deepseek_v4,
            "streaming_parser_engine.py": patch_streaming_engine,
            "adapters.py": patch_adapters,
            "parser_engine.py": patch_parser_engine,
        }
        name = fp.split("/")[-1]
        try:
            patch_fns[name](p[0])
        except SystemExit:
            ok = False
    if not ok:
        print("WARNING: some files could not be patched.", file=sys.stderr)
    print("Patch complete.")


if __name__ == "__main__":
    main()
