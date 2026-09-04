#!/usr/bin/env python3
"""Build isolated variants to separate "CC version" from "conversation shape"
as the cause of the empty-output bug.

variant C: 260's payload but trimmed down to 220's 3-message shape (drop the
  hello/hello exchange and the trailing <total_tokens> system message) --
  keeps 260's version string + tool list.
variant D: 220's payload but with a hello/hello exchange inserted before the
  real question, plus a trailing <total_tokens> system message appended --
  mimics 260's shape while keeping 220's version string + tool list.
"""
import json
import copy

d220 = json.load(open("data/claude-2.1.220-request.openai.json"))
d260 = json.load(open("data/claude-2.1.260-request.openai.json"))


def make_variant_c():
    d = copy.deepcopy(d260)
    msgs = d["messages"]
    # msgs: 0=system(header), 1=user(hello), 2=system(agent types), 3=assistant(hello),
    #       4=user(real question), 5=system(total_tokens)
    new_msgs = [msgs[0], msgs[4], msgs[2]]
    d["messages"] = new_msgs
    return d


def make_variant_d():
    d = copy.deepcopy(d220)
    msgs = d["messages"]
    # msgs: 0=system(header), 1=user(real question, with /clear local-command blocks), 2=system(agent types)
    hello_user = {"role": "user", "content": "hello"}
    hello_assistant = {"role": "assistant", "content": "hello"}
    total_tokens_system = {
        "role": "system",
        "content": [{"type": "text", "text": "<total_tokens>15000000 tokens left</total_tokens>"}],
    }
    new_msgs = [msgs[0], hello_user, msgs[2], hello_assistant, msgs[1], total_tokens_system]
    d["messages"] = new_msgs
    return d


if __name__ == "__main__":
    c = make_variant_c()
    d = make_variant_d()
    json.dump(c, open("data/variant-c-260shape-trimmed.openai.json", "w"), indent=2)
    json.dump(d, open("data/variant-d-220-with-hello.openai.json", "w"), indent=2)
    print("variant C (260 content, 220 shape): num messages =", len(c["messages"]))
    print("variant D (220 content, 260 shape): num messages =", len(d["messages"]))
