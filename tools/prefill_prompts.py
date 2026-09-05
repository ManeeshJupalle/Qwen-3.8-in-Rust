"""Phase 5.5: the prefill timing prompts, 32 / 128 / 512 raw ids (no chat template, no BOS, as tools/prompts.py),
cut from one long English text so the three are prefixes of each other. The text is docs/spec.md as of the
generation (its sha256 is recorded); the ids are what the measurement feeds (`aqueduct run --ids`), so a later
edit of the document does not move the fixture. Writes tests/fixtures/prefill_prompts.json.

Usage: python tools/prefill_prompts.py <path/to/tokenizer.json>
"""
import hashlib
import json
import os
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "prefill_prompts.json")
SOURCE = os.path.join(ROOT, "docs", "spec.md")
LENGTHS = [32, 128, 512]


def main():
    tok_path = sys.argv[1]
    tok = Tokenizer.from_file(tok_path)
    text = open(SOURCE, encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= max(LENGTHS), "source text too short: %d ids" % len(ids)
    out = {
        "tokenizer_json_sha256": hashlib.sha256(open(tok_path, "rb").read()).hexdigest(),
        "source": os.path.relpath(SOURCE, ROOT).replace(os.sep, "/"),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "add_special_tokens": False,
        "bos_prepended": False,
        "prompts": [{"name": "p%d" % n, "n_ids": n, "ids": ids[:n]} for n in LENGTHS],
    }
    for p in out["prompts"]:
        print("%-5s n_ids=%3d first ids=%s" % (p["name"], p["n_ids"], p["ids"][:8]))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
