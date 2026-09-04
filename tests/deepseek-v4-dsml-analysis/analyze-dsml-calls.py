#!/usr/bin/env python3
"""
analyze-dsml-calls.py - Correlate DSML tool calls with their parse outcome

Joins a Claude Code session transcript (.jsonl) with the vLLM server logs
produced by recipes/deepseek-v4-flash-0731-debug.yaml (--enable-log-outputs),
so every DSML tool-call attempt can be reviewed end to end:

    raw model text  ->  DSML structure  ->  parsed result  ->  context size

The server log supplies the raw text the DSML parser actually saw. The
transcript supplies what the client ended up with (tool_use blocks vs. DSML
leaked as plain text) plus the context token count for that request.

Records are matched between the two sources by content signature (the
reasoning/answer text, which is identical on both sides) and fall back to
timestamp proximity.

Usage:
    ./analyze-dsml-calls.py data/<session>.jsonl data/spark1.log data/spark3.log
    ./analyze-dsml-calls.py <session>.jsonl <log...> --failures-only
    ./analyze-dsml-calls.py <session>.jsonl <log...> --detail
    ./analyze-dsml-calls.py <session>.jsonl <log...> --json > dsml.json
    ./analyze-dsml-calls.py <session>.jsonl <log...> --csv dsml.csv
    ./analyze-dsml-calls.py <session>.jsonl <log...> --dump-raw /tmp/dsml-raw
"""

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# DSML tags use fullwidth vertical bars, e.g. <｜DSML｜invoke name="Bash">
BAR = "｜"
DSML_MARK = f"<{BAR}DSML{BAR}"
WRAPPER_TAGS = ("tool_calls", "toolcalls", "function_calls")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESPONSE_RE = re.compile(
    r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d).*?Generated response "
    r"(?P<rid>[^\s(]+) \((?P<phase>[^)]*)\): output: "
)
TAIL_MARK = ", output_token_ids: "
DSML_TAGS = ("tool_calls", "toolcalls", "function_calls", "invoke", "parameter")
# The model mixes fullwidth DSML tags with bare ASCII ones (<parameter ...>),
# often within a single response, so both forms are recognized and counted.
_TAG_ALT = "|".join(DSML_TAGS)
OPEN_TAG_RE = re.compile(rf"<({BAR}DSML{BAR})?({_TAG_ALT})\b([^>]*)>")
CLOSE_TAG_RE = re.compile(rf"</({BAR}DSML{BAR})?({_TAG_ALT})>")
NAME_ATTR_RE = re.compile(r'name="([^"]*)"')
LEADING_CLOSERS_RE = re.compile(rf"(?:</(?:{BAR}DSML{BAR})?(?:{_TAG_ALT})>\s*)+")
PARAM_VALUE_RE = re.compile(
    rf"<(?:{BAR}DSML{BAR})?parameter\b[^>]*>(.*?)</(?:{BAR}DSML{BAR})?parameter>",
    re.DOTALL,
)
WS_RE = re.compile(r"\s+")
SIG_LEN = 200
SIG_MIN = 40


def signature(text):
    """Normalized prefix used to match a log record to a transcript message."""
    return WS_RE.sub(" ", (text or "")).strip()[:SIG_LEN]


def decode_repr(chunk):
    """Turn the Python repr of the output string back into text."""
    try:
        value = ast.literal_eval(chunk)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def parse_server_log(path, year):
    """Extract every logged model output from one vLLM server log."""
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Generated response" not in line:
                continue
            line = ANSI_RE.sub("", line.rstrip("\r\n"))
            match = RESPONSE_RE.search(line)
            if not match:
                continue
            body = line[match.end():]
            # The line ends with ", output_token_ids: ..., finish_reason: ...";
            # everything before the last occurrence is the repr of the output.
            cut = body.rfind(TAIL_MARK)
            finish = ""
            if cut == -1:
                raw_repr = body
            else:
                raw_repr = body[:cut]
                tail = body[cut + len(TAIL_MARK):]
                if "finish_reason: " in tail:
                    finish = tail.split("finish_reason: ", 1)[1].strip().rstrip(".")
            text = decode_repr(raw_repr)
            if text is None:
                continue
            stamp = datetime.strptime(f"{year}-{match.group('ts')}", "%Y-%m-%d %H:%M:%S")
            records.append({
                "request_id": match.group("rid"),
                "phase": match.group("phase"),
                "log": Path(path).name,
                "time": stamp.replace(tzinfo=timezone.utc),
                "raw": text,
                "finish_reason": finish,
            })
    return records


def dsml_segment(text):
    """The part of the output from the first tool-call tag onward.

    Covers both the fullwidth DSML form and the bare ASCII form the model
    sometimes emits instead.
    """
    starts = [m.start() for m in (OPEN_TAG_RE.search(text), CLOSE_TAG_RE.search(text)) if m]
    return "" if not starts else text[min(starts):]


def analyze_dsml(text):
    """Classify the DSML structure of a raw model output."""
    # Closers left over from the previous turn can prefix an otherwise valid
    # response; they are noted but excluded from the structural balance.
    segment = dsml_segment(text)
    leading = LEADING_CLOSERS_RE.match(segment)
    if leading:
        remainder = segment[leading.end():]
        # Only a prefix: keep it, it is the whole story (stray-closers turn).
        if OPEN_TAG_RE.search(remainder) or CLOSE_TAG_RE.search(remainder):
            text = text.replace(leading.group(0), "", 1)
        else:
            leading = None

    opens, self_closing, ascii_tags, fullwidth_tags = Counter(), Counter(), 0, 0
    for prefix, tag, attrs in OPEN_TAG_RE.findall(text):
        (self_closing if attrs.rstrip().endswith("/") else opens)[tag] += 1
        if prefix:
            fullwidth_tags += 1
        else:
            ascii_tags += 1
    closes = Counter()
    for prefix, tag in CLOSE_TAG_RE.findall(text):
        closes[tag] += 1
        if prefix:
            fullwidth_tags += 1
        else:
            ascii_tags += 1

    names = [
        NAME_ATTR_RE.search(attrs).group(1)
        for _, tag, attrs in OPEN_TAG_RE.findall(text)
        if tag == "invoke" and NAME_ATTR_RE.search(attrs)
    ]
    wrapper_open = sum(opens[t] for t in WRAPPER_TAGS)
    wrapper_close = sum(closes[t] for t in WRAPPER_TAGS)
    invoke_open = opens["invoke"] + self_closing["invoke"]
    invoke_close = closes["invoke"]
    any_tag = bool(opens or self_closing or closes)

    param_open = opens["parameter"] + self_closing["parameter"]
    flags = ["leading-closer"] if leading else []
    if invoke_open and not wrapper_open:
        flags.append("orphan-invoke")
    if param_open and not invoke_open:
        flags.append("orphan-parameter")
    if ascii_tags and fullwidth_tags:
        flags.append("mixed-tag-form")
    elif ascii_tags and not fullwidth_tags:
        flags.append("ascii-tags-only")
    if (closes and not opens and not self_closing):
        flags.append("stray-closers")
    if invoke_open > invoke_close:
        flags.append("unclosed-invoke")
    if wrapper_open > wrapper_close:
        flags.append("unclosed-wrapper")
    if wrapper_close > wrapper_open:
        flags.append("extra-wrapper-close")
    if opens["parameter"] > closes["parameter"] + self_closing["parameter"]:
        flags.append("unclosed-parameter")

    if not any_tag:
        shape = "no-dsml"
    elif "stray-closers" in flags:
        shape = "stray-closers"
    elif "orphan-parameter" in flags:
        # No invoke header at all: the parser never enters tool-call state,
        # so the whole block leaks to the client as plain text.
        shape = "orphan-parameter"
    elif "orphan-invoke" in flags:
        shape = "orphan-invoke"
    elif "unclosed-invoke" in flags or "unclosed-wrapper" in flags:
        shape = "truncated"
    elif wrapper_open and invoke_open and param_open >= closes["parameter"]:
        shape = "well-formed"
    else:
        shape = "other"

    return {
        "shape": shape,
        "flags": flags,
        "raw_tool_names": names,
        "wrapper_open": wrapper_open,
        "wrapper_close": wrapper_close,
        "invoke_open": invoke_open,
        "invoke_close": invoke_close,
        "param_open": param_open,
        "param_close": closes["parameter"],
        "ascii_tags": ascii_tags,
        "fullwidth_tags": fullwidth_tags,
        "has_dsml": any_tag,
    }


def load_transcript(path):
    """Group assistant messages of a session by message id.

    Streaming splits one model response across several transcript lines, so
    blocks are merged per message id. Returns messages in first-seen order.
    """
    messages, order = {}, []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message") or {}
            mid = message.get("id")
            if not mid:
                continue
            if mid not in messages:
                order.append(mid)
                messages[mid] = {
                    "id": mid,
                    "time": entry.get("timestamp"),
                    "context_tokens": (message.get("usage") or {}).get("input_tokens"),
                    "output_tokens": (message.get("usage") or {}).get("output_tokens"),
                    "stop_reason": message.get("stop_reason"),
                    "thinking": "",
                    "text": "",
                    "tool_uses": [],
                }
            record = messages[mid]
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "thinking":
                    record["thinking"] += block.get("thinking") or ""
                elif kind == "text":
                    record["text"] += block.get("text") or ""
                elif kind == "tool_use":
                    record["tool_uses"].append({
                        "name": block.get("name"),
                        "input": block.get("input"),
                    })
    return [messages[mid] for mid in order]


def parse_iso(stamp):
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def corroborates(raw, message):
    """True if a candidate transcript message really is this raw output.

    Checks the two independent ways raw text shows up client-side: parameter
    values landing in a tool_use input, or tool-call markup leaking as text.
    """
    values = [v for v in PARAM_VALUE_RE.findall(raw) if len(v.strip()) >= 12]
    if values:
        rendered = json.dumps(
            [u["input"] for u in message["tool_uses"]], ensure_ascii=False
        )
        for value in values[:6]:
            probe = value.strip()[:60]
            if probe and probe in rendered:
                return True
    body = message["text"] + message["thinking"]
    for value in values[:6]:
        probe = value.strip()[:60]
        if probe and probe in body:
            return True

    # Short closer-only outputs carry no parameter values. Fall back to the
    # length relation, which holds tightly across this corpus (~3-4 chars per
    # output token), so a same-second candidate of the wrong size is rejected.
    tokens = message["output_tokens"]
    if not values and tokens and 0 < len(raw) <= 400:
        return 2.0 <= len(raw) / tokens <= 6.0
    return False


def correlate(log_records, messages, window_s):
    """Attach each log record to a transcript message.

    Primary key is the content signature (reasoning first, then answer text),
    which is identical on both sides. Ambiguous signatures are dropped and
    resolved by timestamp proximity instead.
    """
    by_sig, ambiguous = {}, set()
    for message in messages:
        for field in ("thinking", "text"):
            sig = signature(message[field])
            if len(sig) < SIG_MIN:
                continue
            if sig in by_sig and by_sig[sig] is not message:
                ambiguous.add(sig)
            by_sig.setdefault(sig, message)
    for sig in ambiguous:
        by_sig.pop(sig, None)

    for record in log_records:
        raw = record["raw"]
        # Server-side raw text is "<reasoning></think>answer"; the transcript
        # splits the same content into thinking and text blocks.
        head, _, tail = raw.partition("</think>")
        candidates = [signature(head), signature(tail or raw), signature(raw)]
        match, how = None, ""
        for sig in candidates:
            if len(sig) >= SIG_MIN and sig in by_sig:
                match, how = by_sig[sig], "content"
                break
        if match is None:
            # Timestamp alone is not enough: other sessions share the server,
            # so a candidate is only accepted if its content corroborates.
            best = best_weak = None
            for message in messages:
                stamp = parse_iso(message["time"])
                if stamp is None:
                    continue
                delta = abs((stamp - record["time"]).total_seconds())
                if delta > window_s:
                    continue
                if corroborates(raw, message):
                    if best is None or delta < best[0]:
                        best = (delta, message)
                elif best_weak is None or delta < best_weak[0]:
                    best_weak = (delta, message)
            if best:
                match, how = best[1], f"time±{best[0]:.0f}s"
            elif best_weak:
                match, how = best_weak[1], f"time±{best_weak[0]:.0f}s?"
        record["message"] = match
        record["match"] = how or "unmatched"
    return log_records


def build_row(record):
    """Combine raw text, DSML structure and client-side result into one row."""
    raw = record["raw"]
    info = analyze_dsml(raw)
    message = record.get("message")

    tool_names, context, out_tokens = [], None, None
    leaked_text = leaked_reasoning = contaminated = leading_artifact = False
    if message:
        context = message["context_tokens"]
        out_tokens = message["output_tokens"]
        # A leak is any tool-call tag reaching the client as plain text, in
        # either the fullwidth DSML form or the bare ASCII form.
        body = message["text"]
        leaked_text = bool(OPEN_TAG_RE.search(body) or CLOSE_TAG_RE.search(body))
        # A turn can open with stray closers left over from the previous turn
        # while its own call still parses. That is cosmetic, not a lost call,
        # so it is tracked separately from a real leak.
        stripped = LEADING_CLOSERS_RE.sub("", body.lstrip(), count=1)
        leading_artifact = leaked_text and not (
            OPEN_TAG_RE.search(stripped) or CLOSE_TAG_RE.search(stripped)
        )
        if leading_artifact:
            leaked_text = False
        # A call swallowed into the reasoning block never reaches the client
        # either, so it counts as its own failure mode.
        leaked_reasoning = bool(
            OPEN_TAG_RE.search(message["thinking"])
            or CLOSE_TAG_RE.search(message["thinking"])
        )
        for use in message["tool_uses"]:
            tool_names.append(use["name"])
            rendered = json.dumps(use["input"], ensure_ascii=False)
            if OPEN_TAG_RE.search(rendered) or CLOSE_TAG_RE.search(rendered):
                contaminated = True

    expected = len(info["raw_tool_names"])
    if message is None:
        parse = "unknown"
    elif contaminated:
        parse = "contaminated"
    elif leaked_text and tool_names:
        # Some of the intended calls parsed, the rest leaked as text.
        parse = "partial"
    elif leaked_text:
        parse = "leaked"
    elif leaked_reasoning and not tool_names:
        parse = "leaked-reasoning"
    elif tool_names and expected and len(tool_names) < expected:
        parse = "partial"
    elif tool_names:
        parse = "ok"
    else:
        parse = "dropped"

    row = dict(record)
    row.pop("message", None)
    row.update(info)
    row.update({
        "parse": parse,
        "success": parse == "ok",
        "context_tokens": context,
        "output_tokens": out_tokens,
        "parsed_tool_names": tool_names,
        "leaked_text": leaked_text,
        "leaked_reasoning": leaked_reasoning,
        "leading_artifact": leading_artifact,
        "contaminated_args": contaminated,
        "verified": not record["match"].endswith("?"),
        "message_id": message["id"] if message else None,
        "dsml": dsml_segment(raw),
        "raw_len": len(raw),
    })
    return row


def fmt_tokens(value):
    return "-" if value is None else f"{value:,}"


def print_table(rows):
    header = (
        f"{'#':>4}  {'time':<8}  {'request_id':<26}  {'ctx':>8}  {'out':>6}  "
        f"{'shape':<16}  {'parse':<16}  tools"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        raw_names = ",".join(row["raw_tool_names"]) or "-"
        got_names = ",".join(row["parsed_tool_names"]) or "-"
        tools = raw_names if raw_names == got_names else f"{raw_names} -> {got_names}"
        mark = "" if row["verified"] else " ?"
        print(
            f"{row['seq']:>4}  {row['time'].strftime('%H:%M:%S'):<8}  "
            f"{row['request_id']:<26}  {fmt_tokens(row['context_tokens']):>8}  "
            f"{fmt_tokens(row['output_tokens']):>6}  {row['shape']:<16}  "
            f"{row['parse']:<16}  {tools}{mark}"
        )
    if any(not r["verified"] for r in rows):
        print("\n? = matched by timestamp only, content could not corroborate it")


def print_detail(rows, excerpt):
    for row in rows:
        print("=" * 100)
        print(
            f"#{row['seq']}  {row['request_id']}  {row['log']}  "
            f"{row['time'].strftime('%Y-%m-%d %H:%M:%S')}Z"
        )
        print(
            f"  context={fmt_tokens(row['context_tokens'])} tokens   "
            f"output={fmt_tokens(row['output_tokens'])} tokens   "
            f"raw_len={row['raw_len']}   finish={row['finish_reason'] or '-'}   "
            f"match={row['match']}"
        )
        print(
            f"  shape={row['shape']}   flags={','.join(row['flags']) or '-'}   "
            f"wrapper={row['wrapper_open']}/{row['wrapper_close']}   "
            f"invoke={row['invoke_open']}/{row['invoke_close']}"
        )
        print(f"  parse={row['parse']}   success={row['success']}")
        print(f"  raw tool names   : {', '.join(row['raw_tool_names']) or '-'}")
        print(f"  parsed tool names: {', '.join(row['parsed_tool_names']) or '-'}")
        segment = row["dsml"]
        shown = segment[:excerpt] if excerpt else segment
        print(f"  --- raw DSML ({len(segment)} chars"
              f"{', truncated' if len(shown) < len(segment) else ''}) ---")
        for line in shown.splitlines() or [""]:
            print(f"  | {line}")
    print("=" * 100)


def print_summary(rows, total_logged, unmatched):
    ok = sum(1 for r in rows if r["success"])
    print(f"\nDSML tool-call attempts : {len(rows)}"
          f"   (of {total_logged} logged responses)")
    if unmatched:
        print(f"unmatched in transcript : {unmatched}"
              f"  (other sessions sharing the server, or truncated log)")
    if rows:
        print(f"parsed successfully     : {ok}/{len(rows)}"
              f"  ({100.0 * ok / len(rows):.1f}%)")
    for title, key in (("by parse outcome", "parse"), ("by DSML shape", "shape")):
        counts = Counter(r[key] for r in rows)
        print(f"\n{title}:")
        for name, count in counts.most_common():
            print(f"  {name:<16} {count:>4}")

    def median(values):
        values = sorted(v for v in values if v)
        if not values:
            return 0
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    groups = [("success", [r for r in rows if r["success"]])]
    for shape in sorted({r["shape"] for r in rows if not r["success"]}):
        groups.append((shape, [r for r in rows if not r["success"] and r["shape"] == shape]))
    if len(groups) > 1:
        print("\ncontext tokens by outcome (n / median / min / max):")
        for name, group in groups:
            values = [r["context_tokens"] for r in group if r["context_tokens"]]
            if not values:
                print(f"  {name:<18} {len(group):>4}          -")
                continue
            print(f"  {name:<18} {len(group):>4}  {median(values):>9,.0f}"
                  f"  {min(values):>9,}  {max(values):>9,}")

    buckets = Counter()
    for row in rows:
        context = row["context_tokens"]
        if not context:
            continue
        bucket = int(context // 25000) * 25
        buckets[bucket] += 0 if row["success"] else 1
        buckets[(bucket, "n")] = buckets.get((bucket, "n"), 0) + 1
    if buckets:
        print("\nfailure rate by context size:")
        for bucket in sorted(b for b in buckets if isinstance(b, int)):
            total = buckets[(bucket, "n")]
            bad = buckets[bucket]
            bar = "#" * int(round(20.0 * bad / total)) if total else ""
            print(f"  {bucket:>4}k-{bucket + 25:<4}k  {bad:>3}/{total:<4}"
                  f" {100.0 * bad / total:>5.1f}%  {bar}")


def write_csv(rows, path):
    fields = [
        "seq", "request_id", "log", "timestamp", "match", "context_tokens",
        "output_tokens", "raw_len", "shape", "flags", "parse", "success",
        "raw_tool_names", "parsed_tool_names", "leaked_text",
        "contaminated_args", "finish_reason", "message_id", "dsml", "raw",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["timestamp"] = row["time"].isoformat()
            for key in ("flags", "raw_tool_names", "parsed_tool_names"):
                out[key] = ",".join(row[key])
            writer.writerow(out)


def dump_raw(rows, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = f"{row['seq']:04d}-{row['parse']}-{row['shape']}-{row['request_id']}.txt"
        (target / name).write_text(row["raw"], encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Correlate DSML tool calls with their parse outcome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session", help="Claude Code session transcript (.jsonl)")
    parser.add_argument("logs", nargs="+", help="vLLM server log(s) with --enable-log-outputs")
    parser.add_argument("--failures-only", action="store_true",
                        help="only rows whose tool call did not parse cleanly")
    parser.add_argument("--detail", action="store_true",
                        help="print raw DSML and parsed result per record")
    parser.add_argument("--excerpt", type=int, default=400, metavar="N",
                        help="chars of raw DSML to show in --detail (0 = all, default 400)")
    parser.add_argument("--all-responses", action="store_true",
                        help="include responses with no DSML at all")
    parser.add_argument("--include-unmatched", action="store_true",
                        help="keep log records absent from the transcript (other sessions)")
    parser.add_argument("--window", type=float, default=90.0, metavar="SEC",
                        help="timestamp fallback window when content match fails (default 90)")
    parser.add_argument("--year", type=int, default=None,
                        help="year for the year-less server log timestamps")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument("--csv", metavar="PATH", help="also write a CSV table")
    parser.add_argument("--dump-raw", metavar="DIR",
                        help="write each raw model output to its own file")
    args = parser.parse_args()

    messages = load_transcript(args.session)
    if not messages:
        sys.exit(f"no assistant messages found in {args.session}")

    year = args.year
    if year is None:
        stamps = [parse_iso(m["time"]) for m in messages]
        stamps = [s for s in stamps if s]
        year = stamps[0].year if stamps else datetime.now(timezone.utc).year

    records = []
    for log in args.logs:
        found = parse_server_log(log, year)
        if not found:
            print(f"warning: no logged responses in {log}", file=sys.stderr)
        records.extend(found)

    seen, deduped = set(), []
    for record in sorted(records, key=lambda r: r["time"]):
        key = (record["request_id"], record["raw"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)

    correlate(deduped, messages, args.window)
    unmatched = sum(1 for r in deduped if r["match"] == "unmatched")
    if not args.include_unmatched:
        deduped = [r for r in deduped if r["match"] != "unmatched"]

    rows = [build_row(r) for r in deduped]
    if not args.all_responses:
        rows = [r for r in rows if r["has_dsml"]]
    if args.failures_only:
        rows = [r for r in rows if not r["success"]]
    for index, row in enumerate(rows, 1):
        row["seq"] = index

    if args.json:
        print(json.dumps(
            [{**r, "time": r["time"].isoformat()} for r in rows],
            ensure_ascii=False, indent=2,
        ))
    elif args.detail:
        print_detail(rows, args.excerpt)
        print_summary(rows, len(records), unmatched)
    else:
        print_table(rows)
        print_summary(rows, len(records), unmatched)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV written to {args.csv}", file=sys.stderr)
    if args.dump_raw:
        target = dump_raw(rows, args.dump_raw)
        print(f"raw outputs written to {target}/", file=sys.stderr)


if __name__ == "__main__":
    main()
