#!/usr/bin/env python3
"""Replay a converted OpenAI-format request (from convert_anthropic_to_openai.py)
against a live vLLM server, N times, reporting whether each attempt returned
empty content/tool_calls (the "genuine empty output" bug) or a well-formed
response.
"""
import json
import os
import sys
import time

import requests

URL = f"http://{os.environ.get('REPLAY_HOST', '192.168.4.194')}:8000/v1/chat/completions"


def load_payload(path, stream):
    data = json.load(open(path))
    data = dict(data)
    data["stream"] = stream
    data.pop("stream_options", None)
    return data


def run_once(data, stream):
    t0 = time.time()
    if stream:
        resp = requests.post(URL, json=data, timeout=300, stream=True)
        if resp.status_code != 200:
            return {"status": resp.status_code, "error": resp.text[:500]}
        content = ""
        reasoning = ""
        tool_calls_seen = False
        finish_reason = None
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode()
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
            if delta.get("tool_calls"):
                tool_calls_seen = True
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        dt = time.time() - t0
        return {
            "status": 200,
            "elapsed": round(dt, 2),
            "finish_reason": finish_reason,
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "tool_calls_seen": tool_calls_seen,
            "empty": (len(content) == 0 and len(reasoning) == 0 and not tool_calls_seen),
        }
    else:
        resp = requests.post(URL, json=data, timeout=300)
        dt = time.time() - t0
        if resp.status_code != 200:
            return {"status": resp.status_code, "error": resp.text[:500]}
        j = resp.json()
        choice = j["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        tool_calls = msg.get("tool_calls") or []
        return {
            "status": 200,
            "elapsed": round(dt, 2),
            "finish_reason": choice.get("finish_reason"),
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "tool_calls_seen": len(tool_calls) > 0,
            "tool_calls_empty_args": [
                tc["function"]["arguments"] for tc in tool_calls
                if tc.get("function", {}).get("arguments") in ("{}", "", None)
            ],
            "empty": (len(content) == 0 and len(reasoning) == 0 and not tool_calls),
        }


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    stream = "--stream" in sys.argv

    data = load_payload(path, stream)
    print(f"Replaying {path} against {URL} x{n} (stream={stream})")
    empties = 0
    for i in range(n):
        try:
            result = run_once(data, stream)
        except Exception as e:
            result = {"error": str(e)}
        tag = "EMPTY" if result.get("empty") else ("ERR" if "error" in result else "ok")
        if result.get("empty"):
            empties += 1
        print(f"[{i+1}/{n}] {tag} {result}")
    print(f"\n{empties}/{n} empty")


if __name__ == "__main__":
    main()
