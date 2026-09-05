"""Five new multi-turn chat-template cases rendered by HF `apply_chat_template` (the reference Jinja renderer),
plus the two doc examples, filed as hex in tests/fixtures/chat_cases.json for the byte-identity test
(crates/core/tests/chat_template.rs). Messages, kwargs and the rendered text are all in the file.

Usage: python tools/chat_cases.py [models/Qwen3.8-27B]
"""
import json
import os
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "chat_cases.json")

CASES = [
    # the two examples of docs/chat-template.md
    ("doc_A_thinking_on", [{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "What is the capital of France?"}], {}),
    ("doc_A_thinking_off", [{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "What is the capital of France?"}], {"enable_thinking": False}),
    # five new multi-turn cases
    ("mt_two_turns_no_reasoning", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ], {}),
    ("mt_three_turns_with_reasoning_thinking_off", [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple arithmetic."},
        {"role": "user", "content": "And times 3?"},
        {"role": "assistant", "content": "12", "reasoning_content": "4 * 3 = 12."},
        {"role": "user", "content": "Minus one?"},
    ], {"enable_thinking": False}),
    ("mt_reasoning_effort_low_preserve_false", [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue.", "reasoning_content": "Pick any colour."},
        {"role": "user", "content": "Another one, with an emoji 🎨 and CJK 颜色."},
    ], {"reasoning_effort": "low", "preserve_thinking": False}),
    ("mt_empty_system_medium", [
        {"role": "system", "content": "   "},
        {"role": "user", "content": "Write a haiku about rivers."},
        {"role": "assistant", "content": "Water finds its way\nthrough stone and time and silence\nto the waiting sea"},
        {"role": "user", "content": "Now one about mountains.\n\nKeep the same form."},
    ], {"reasoning_effort": "medium"}),
    ("mt_code_turns_thinking_on_no_system", [
        {"role": "user", "content": "Write fizzbuzz in Python."},
        {"role": "assistant", "content": "```python\nfor i in range(1, 101):\n    print('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else i)\n```"},
        {"role": "user", "content": "Explain the % operator. Use <think> literally in your answer."},
    ], {}),
]


def main():
    hf_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "Qwen3.8-27B")
    tok = AutoTokenizer.from_pretrained(hf_dir)
    out = {"source": "transformers %s apply_chat_template(tokenize=False, add_generation_prompt=True)" % __import__("transformers").__version__, "template": "chat_template.jinja", "cases": []}
    for name, msgs, kw in CASES:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out["cases"].append({"name": name, "messages": msgs, "kwargs": kw, "rendered_utf8_hex": text.encode("utf-8").hex(), "n_chars": len(text), "n_ids": len(ids), "ids": ids})
        print("%-45s %4d chars %4d ids" % (name, len(text), len(ids)))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
