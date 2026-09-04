#!/usr/bin/env python3
import ast
import json
import sys
import requests

RAW_PATH = "data/request.txt"
URL = "http://192.168.4.194:8000/v1/chat/completions"

import os
if os.environ.get("REPLAY_HOST"):
    URL = f"http://{os.environ['REPLAY_HOST']}:8000/v1/chat/completions"

def extract_payload(raw_path):
    text = open(raw_path).read()
    idx = text.index("-d '")
    payload = text[idx + 4:].rstrip()
    if payload.endswith("'"):
        payload = payload[:-1]
    return payload

def main():
    payload_repr = extract_payload(RAW_PATH)
    data = ast.literal_eval(payload_repr)
    data["stream"] = False
    data.pop("stream_options", None)

    if "--stream" in sys.argv:
        data["stream"] = True
        resp = requests.post(URL, json=data, timeout=300, stream=True)
        print("HTTP status:", resp.status_code)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode()
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line.strip() == "[DONE]":
                break
            print(line)
        return

    resp = requests.post(URL, json=data, timeout=300)
    print("HTTP status:", resp.status_code)
    try:
        j = resp.json()
    except Exception:
        print("Non-JSON response:")
        print(resp.text[:2000])
        return

    if resp.status_code != 200:
        print(json.dumps(j, indent=2)[:3000])
        return

    choice = j["choices"][0]
    msg = choice["message"]
    print("finish_reason:", choice.get("finish_reason"))
    print("content repr:", repr(msg.get("content")))
    print("reasoning_content len:", len(msg.get("reasoning_content") or ""))
    print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=2))
    print("usage:", json.dumps(j.get("usage"), indent=2))

if __name__ == "__main__":
    main()
