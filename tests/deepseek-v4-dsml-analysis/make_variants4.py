#!/usr/bin/env python3
"""Precise test of the hypothesis: empty output requires BOTH
(a) the conversation already contains one completed assistant turn, AND
(b) the request ends on a system-role message.
Neither alone (per variants G/H) reproduced it; full 260 (both present) did.

260 messages: 0=system(header), 1=user(hello-context), 2=system(agent-types, mid),
              3=assistant("hello"), 4=user(real question), 5=system(trailing marker)
"""
import json
import copy

d260 = json.load(open("data/claude-2.1.260-request.openai.json"))
msgs = d260["messages"]


def variant_j_no_mid_system():
    """Remove the mid-conversation system message (msgs[2]) but keep
    hello pair (assistant turn present) AND trailing marker. Tests
    whether msgs[2] specifically is required, or just any assistant-turn
    + trailing-system pattern.
    """
    d = copy.deepcopy(d260)
    d["messages"] = [msgs[0], msgs[1], msgs[3], msgs[4], msgs[5]]
    return d


if __name__ == "__main__":
    j = variant_j_no_mid_system()
    json.dump(j, open("data/variant-j-260-no-mid-system.openai.json", "w"), indent=2)
    print("J (260 real, drop mid system msg, keep hello+trailing): num messages =", len(j["messages"]))
    for i, m in enumerate(j["messages"]):
        print(" ", i, m["role"])
