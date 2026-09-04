#!/usr/bin/env python3
"""Variant K: same as J (assistant turn earlier + trailing marker) but the
trailing marker is sent as role=user instead of role=system, to test whether
it's specifically the system role (vs. just "one more message after the
real question") that matters.
"""
import json
import copy

d260 = json.load(open("data/claude-2.1.260-request.openai.json"))
msgs = d260["messages"]


def variant_k_trailing_as_user():
    d = copy.deepcopy(d260)
    trailing_as_user = copy.deepcopy(msgs[5])
    trailing_as_user["role"] = "user"
    d["messages"] = [msgs[0], msgs[1], msgs[3], msgs[4], trailing_as_user]
    return d


if __name__ == "__main__":
    k = variant_k_trailing_as_user()
    json.dump(k, open("data/variant-k-trailing-as-user.openai.json", "w"), indent=2)
    print("K (same as J but trailing role=user): num messages =", len(k["messages"]))
    for i, m in enumerate(k["messages"]):
        print(" ", i, m["role"])
