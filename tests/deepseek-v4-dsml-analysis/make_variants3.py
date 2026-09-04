#!/usr/bin/env python3
"""Subtractive variants built directly from 260's REAL messages (not
reconstructed from 220), to isolate which piece of 260's actual content
drives the high empty rate.

260 messages: 0=system(header), 1=user(sys-reminder+/model-switch+hello),
              2=system(agent types), 3=assistant("hello"),
              4=user(real question), 5=system(total_tokens)
"""
import json
import copy

d260 = json.load(open("data/claude-2.1.260-request.openai.json"))
msgs = d260["messages"]


def variant_g_no_trailing_marker():
    """Drop only the trailing total_tokens system message (5 msgs)."""
    d = copy.deepcopy(d260)
    d["messages"] = msgs[:5]
    return d


def variant_h_no_hello_pair():
    """Drop the hello user/assistant pair (msgs[1],[3]), keep everything else (4 msgs)."""
    d = copy.deepcopy(d260)
    d["messages"] = [msgs[0], msgs[2], msgs[4], msgs[5]]
    return d


def variant_i_no_hello_no_trailing():
    """Drop both the hello pair and the trailing marker (3 msgs: header, agent-types, question)."""
    d = copy.deepcopy(d260)
    d["messages"] = [msgs[0], msgs[2], msgs[4]]
    return d


if __name__ == "__main__":
    g = variant_g_no_trailing_marker()
    h = variant_h_no_hello_pair()
    i = variant_i_no_hello_no_trailing()
    json.dump(g, open("data/variant-g-260-no-trailing.openai.json", "w"), indent=2)
    json.dump(h, open("data/variant-h-260-no-hello.openai.json", "w"), indent=2)
    json.dump(i, open("data/variant-i-260-no-hello-no-trailing.openai.json", "w"), indent=2)
    print("G (260 real, no trailing marker): num messages =", len(g["messages"]))
    print("H (260 real, no hello pair): num messages =", len(h["messages"]))
    print("I (260 real, no hello, no trailing): num messages =", len(i["messages"]))
