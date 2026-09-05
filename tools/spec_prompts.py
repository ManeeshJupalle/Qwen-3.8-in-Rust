"""The six prompts of the Phase 5.4 acceptance measurement: the three fixture prompts (raw ids, no template) and
three chat-style prompts (a code request, a factual question, a short essay ask) rendered by HF
`apply_chat_template` with thinking OFF, so the 200 measured tokens are the answer (code, facts, prose)
rather than reasoning. Writes tests/fixtures/spec_prompts.json.

Usage: python tools/spec_prompts.py [models/Qwen3.8-27B]
"""
import json
import os
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "spec_prompts.json")

CHAT = [
    ("code", "Write a Python function that parses an ISO-8601 date string and returns the day of the week. Include a short docstring and two examples."),
    ("fact", "What causes the seasons on Earth, and why are the seasons reversed between the northern and southern hemispheres?"),
    ("essay", "Write a short essay (three paragraphs) on why rivers were central to the growth of early cities."),
]


def main():
    hf_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "Qwen3.8-27B")
    tok = AutoTokenizer.from_pretrained(hf_dir)
    fixture = json.load(open(os.path.join(ROOT, "tests", "fixtures", "prompts.json"), encoding="utf-8"))["prompts"]
    out = {"note": "fixture prompts are raw ids (no template); chat prompts are rendered with enable_thinking=False and add_generation_prompt=True", "prompts": []}
    for p in fixture:
        out["prompts"].append({"name": p["name"], "kind": "fixture", "text": p["text"], "ids": p["ids"], "n_ids": len(p["ids"])})
    for name, text in CHAT:
        rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        out["prompts"].append({"name": name, "kind": "chat (thinking off)", "text": text, "rendered": rendered, "ids": ids, "n_ids": len(ids)})
    for p in out["prompts"]:
        print("%-9s %-20s %4d ids" % (p["name"], p["kind"], p["n_ids"]))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
