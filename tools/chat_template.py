"""Render the chat template with thinking on and off and write docs/chat-template.md.

Uses transformers' AutoTokenizer.apply_chat_template (the reference Jinja renderer) on the
downloaded tokenizer files. Also tokenizes each rendering and lists the special token ids involved.

Usage: python tools/chat_template.py <hf_local_dir>
"""
import json
import os
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "chat-template.md")

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
MESSAGES_NOSYS = [{"role": "user", "content": "What is the capital of France?"}]
MULTI_TURN = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?", "reasoning_content": "The user greeted me."},
    {"role": "user", "content": "Tell me a joke."},
]


def fence(s):
    return "```\n" + s + "\n```"


def main():
    hf_dir = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(hf_dir)
    raw = open(os.path.join(hf_dir, "chat_template.jinja"), encoding="utf-8").read()
    cfg_tpl = json.load(open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8")).get("chat_template")

    def render(msgs, **kw):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def ids_of(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    variants = [
        ("thinking ON (default: enable_thinking undefined -> reasoning_effort defaults to xhigh)", MESSAGES, {}),
        ("thinking ON, reasoning_effort=medium", MESSAGES, {"reasoning_effort": "medium"}),
        ("thinking ON, reasoning_effort=low", MESSAGES, {"reasoning_effort": "low"}),
        ("thinking OFF (enable_thinking=False)", MESSAGES, {"enable_thinking": False}),
        ("thinking ON, no system message", MESSAGES_NOSYS, {}),
        ("thinking OFF, no system message", MESSAGES_NOSYS, {"enable_thinking": False}),
        ("multi-turn with prior reasoning_content, thinking ON (preserve_thinking default)", MULTI_TURN, {}),
        ("multi-turn with prior reasoning_content, preserve_thinking=False", MULTI_TURN, {"preserve_thinking": False}),
    ]

    L = []
    P = L.append
    P("# Chat template (captured from `chat_template.jinja`, rendered by transformers %s)" % __import__("transformers").__version__)
    P("")
    P("Source file: `models/Qwen3.8-27B/chat_template.jinja` (%d chars). The same template is also embedded in" % len(raw))
    P("`tokenizer_config.json:chat_template` (identical: **%s**)." % (raw == cfg_tpl))
    P("")
    P("## Facts the engine needs")
    P("")
    P("- Template kwargs recognised: `enable_thinking` (undefined or true = ON), `reasoning_effort` (`xhigh` default, `medium`, `low`;")
    P("  anything else raises), `preserve_thinking` (undefined or true = keep `<think>` blocks of ALL prior assistant turns),")
    P("  `add_vision_id`, `tools`, `add_generation_prompt`.")
    P("- Thinking ON: the generation prompt ends with `<|im_start|>assistant\\n<think>\\n` and the model writes the reasoning,")
    P("  then `</think>\\n\\n`, then the answer. With `xhigh` (default) or `low`, a system message containing the reasoning")
    P("  instructions is injected (prepended to the user's system content, or created if there is none). `medium` injects nothing.")
    P("- Thinking OFF (`enable_thinking=False`): the generation prompt ends with `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`,")
    P("  i.e. an EMPTY think block is pre-filled by the template, and NO reasoning-instruction system message is injected.")
    P("- The template never emits a BOS token; `tokenizer_config.json:add_bos_token` is false. `<|im_end|>` (248046) ends every turn.")
    P("- `<think>` (248068) and `</think>` (248069) are added tokens with `special=false` in tokenizer_config.json, but the HF")
    P("  tokenizer still maps the literal text to these single ids (see tests/fixtures/tok_cases.json `special_lookalike`).")
    P("- Tool calls use a `<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>` text format, not JSON.")
    P("- Tool results are wrapped as a `user` turn containing `<tool_response>...</tool_response>`.")
    P("")
    P("## Special token ids involved")
    P("")
    P("| token | id | source |")
    P("|---|---|---|")
    for t in ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>", "<tool_call>", "</tool_call>",
              "<tool_response>", "</tool_response>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]:
        P("| `%s` | %d | tokenizer.convert_tokens_to_ids |" % (t, tok.convert_tokens_to_ids(t)))
    P("| eos_token (`%s`) | %s | tokenizer.eos_token_id |" % (tok.eos_token, tok.eos_token_id))
    P("| pad_token (`%s`) | %s | tokenizer.pad_token_id |" % (tok.pad_token, tok.pad_token_id))
    P("| bos_token | %s | tokenizer.bos_token_id (none) |" % tok.bos_token_id)
    P("")
    P("`generation_config.json:eos_token_id` = `[248046, 248044]` (both `<|im_end|>` and `<|endoftext|>` stop generation).")
    P("")
    P("## Rendered examples")
    P("")
    P("Input messages (A): `%s`" % json.dumps(MESSAGES))
    P("")
    P("Input messages (B, no system): `%s`" % json.dumps(MESSAGES_NOSYS))
    P("")
    P("Input messages (C, multi-turn): `%s`" % json.dumps(MULTI_TURN))
    P("")
    for title, msgs, kw in variants:
        text = render(msgs, **kw)
        ids = ids_of(text)
        P("### %s" % title)
        P("")
        P("kwargs: `%s`" % json.dumps(kw))
        P("")
        P(fence(text))
        P("")
        P("token ids (%d): `%s`" % (len(ids), ids))
        P("")
        P("last 6 tokens decoded: `%s`" % json.dumps([tok.decode([i]) for i in ids[-6:]]))
        P("")
    P("## Raw template")
    P("")
    P("```jinja")
    P(raw)
    P("```")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", OUT, len(L), "lines")
    for title, msgs, kw in variants[:1] + variants[3:4]:
        print("---", title)
        print(render(msgs, **kw))


if __name__ == "__main__":
    main()
