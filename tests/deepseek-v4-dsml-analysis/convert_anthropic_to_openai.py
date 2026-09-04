#!/usr/bin/env python3
"""Convert a captured litellm-proxy Anthropic /v1/messages request into the
OpenAI chat/completions payload litellm would actually send to vLLM, using
litellm's own AnthropicAdapter.translate_completion_input_params_with_tool_mapping.
"""
import json
import sys

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)

LITELLM_PROXY_ONLY_KEYS = {
    "litellm_call_id",
    "litellm_metadata",
    "litellm_trace_id",
    "litellm_session_id",
    "context_management",
    "litellm_logging_obj",
}


def convert(raw_path):
    data = json.load(open(raw_path))
    kwargs = {k: v for k, v in data.items() if k not in LITELLM_PROXY_ONLY_KEYS}
    openai_request, tool_name_mapping = (
        AnthropicAdapter().translate_completion_input_params_with_tool_mapping(kwargs)
    )
    return openai_request, tool_name_mapping


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/claude-2.1.220-request.txt"
    openai_request, tool_name_mapping = convert(raw_path)
    out_path = raw_path.rsplit(".", 1)[0] + ".openai.json"
    with open(out_path, "w") as f:
        json.dump(openai_request, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print("tool_name_mapping:", tool_name_mapping)
    print("keys:", list(openai_request.keys()))
    print("num messages:", len(openai_request.get("messages", [])))
    print("num tools:", len(openai_request.get("tools", []) or []))
    print("stream:", openai_request.get("stream"))


if __name__ == "__main__":
    main()
