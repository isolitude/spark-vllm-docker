#!/usr/bin/env python3
"""Isolate which piece of the 6-message shape triggers empty output:
the hello/hello exchange, or the trailing <total_tokens> system message,
or both together. Base content = 220's (system header + real question + agent-types).
"""
import json
import copy

d220 = json.load(open("data/claude-2.1.220-request.openai.json"))

hello_user = {"role": "user", "content": "hello"}
hello_assistant = {"role": "assistant", "content": "hello"}
total_tokens_system = {
    "role": "system",
    "content": [{"type": "text", "text": "<total_tokens>15000000 tokens left</total_tokens>"}],
}


def base_msgs():
    msgs = d220["messages"]
    return msgs[0], msgs[1], msgs[2]  # system-header, user-question, system-agent-types


def variant_e_hello_only():
    """3-msg base + hello/hello inserted before the real question (no trailing total_tokens)."""
    d = copy.deepcopy(d220)
    sys_header, user_q, sys_agents = base_msgs()
    d["messages"] = [sys_header, hello_user, sys_agents, hello_assistant, user_q]
    return d


def variant_f_total_tokens_only():
    """3-msg base + trailing total_tokens system message (no hello exchange)."""
    d = copy.deepcopy(d220)
    sys_header, user_q, sys_agents = base_msgs()
    d["messages"] = [sys_header, user_q, sys_agents, total_tokens_system]
    return d


if __name__ == "__main__":
    e = variant_e_hello_only()
    f = variant_f_total_tokens_only()
    json.dump(e, open("data/variant-e-hello-only.openai.json", "w"), indent=2)
    json.dump(f, open("data/variant-f-total-tokens-only.openai.json", "w"), indent=2)
    print("variant E (hello/hello, no trailing marker): num messages =", len(e["messages"]))
    print("variant F (trailing marker, no hello): num messages =", len(f["messages"]))
