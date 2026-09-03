"""Tokenizer parity cases. Encodes 45 short strings with HF `tokenizers` (tokenizer.json only,
no transformers wrapper, add_special_tokens=False) and writes tests/fixtures/tok_cases.json as
{text_utf8_hex, ids}. Hex, not raw strings, so nothing is re-encoded by shell or filesystem.

Usage: python tools/tok_cases.py <path/to/tokenizer.json>
"""
import hashlib
import json
import os
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "tok_cases.json")

# label, text. 45 cases. Keep each short; the point is coverage of pre-tokenizer branches.
CASES = [
    # ASCII / English
    ("ascii_simple", "Hello, world!"),
    ("ascii_sentence", "The quick brown fox jumps over the lazy dog."),
    ("ascii_contractions", "I'm sure they've said it'll work, but we're not."),
    ("ascii_numbers", "In 2024 there were 1234567 events, 3.14159 avg, 1e-5 tol."),
    ("ascii_caps_mixed", "NASA vs. nasa vs. NaSa: HTTPServer, iPhone, macOS"),
    ("ascii_punct_run", "Wait... what?!?! -- really; (yes) [no] {maybe} <ok>"),
    ("ascii_url_email", "See https://example.com/a?b=1&c=2#frag or mail me@example.org"),
    ("empty", ""),
    ("single_space", " "),
    ("single_char", "a"),
    # code
    ("code_python", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"),
    ("code_rust", "fn main() {\n    let v: Vec<u8> = vec![0u8; 16];\n    println!(\"{:?}\", &v[..4]);\n}\n"),
    ("code_c_pointers", "int *p = &arr[0]; p += 2; *p |= 0x7F >> 1;"),
    ("code_tabs", "\tif (x)\n\t\treturn y;\n"),
    ("code_shell", "ls -la | grep -E '^d' && echo \"$HOME\" 2>/dev/null"),
    # JSON
    ("json_flat", "{\"key\": \"value\", \"n\": 42, \"ok\": true, \"none\": null}"),
    ("json_nested", "{\"a\":{\"b\":[1,2,{\"c\":\"d\"}]},\"e\":[]}"),
    ("json_escapes", "{\"s\": \"line\\nbreak \\\"quoted\\\" \\u00e9\"}"),
    # Han (Chinese)
    ("han_simplified", "今天天气很好，我们去公园散步吧。"),
    ("han_traditional", "臺灣的夜市小吃非常有名。"),
    ("han_mixed_ascii", "我有3个apple和2只cat。"),
    ("han_classical", "學而時習之，不亦說乎？"),
    # Japanese
    ("ja_hiragana_kanji", "私は毎朝コーヒーを飲みます。"),
    ("ja_katakana", "コンピューターとインターネット"),
    ("ja_mixed_fullwidth", "価格は１２３４円です！"),
    # Korean
    ("ko_hangul", "안녕하세요, 만나서 반갑습니다."),
    ("ko_mixed", "서울의 인구는 약 950만 명입니다."),
    # Cyrillic
    ("ru_sentence", "Привет, как дела? Всё хорошо."),
    ("uk_sentence", "Київ — столиця України."),
    # Arabic (RTL)
    ("ar_sentence", "مرحبا بالعالم، كيف حالك اليوم؟"),
    ("ar_with_digits", "السعر ٤٥٠ ريال أو 450 SAR"),
    # emoji, ZWJ, skin tones, flags
    ("emoji_basic", "I love 🍕 and 🎉!"),
    ("emoji_zwj_family", "👨‍👩‍👧‍👦 family"),
    ("emoji_zwj_profession", "👩🏽‍💻 coder 🧑🏿‍🚀"),
    ("emoji_flags_keycap", "🇯🇵🇺🇸 1️⃣2️⃣"),
    ("emoji_variation_selector", "❤️ vs ❤ vs ☺️"),
    # whitespace runs
    ("ws_multi_spaces", "a    b        c"),
    ("ws_tabs_spaces", "a \t b\t\t c"),
    ("ws_only_run", "        "),
    ("ws_nbsp_ideographic", "a\u00a0b\u3000c"),
    # leading / trailing newlines
    ("nl_leading", "\n\nleading newlines"),
    ("nl_trailing", "trailing newlines\n\n\n"),
    ("nl_crlf", "line1\r\nline2\r\n"),
    ("nl_only", "\n"),
    # special-token-looking text (must be tokenized as text, add_special_tokens=False)
    ("special_lookalike", "<|im_start|>user\n<think>\nhi</think>"),
]


def main():
    tok_path = sys.argv[1]
    tok = Tokenizer.from_file(tok_path)
    assert len(CASES) == 45, len(CASES)
    labels = [c[0] for c in CASES]
    assert len(set(labels)) == 45, "duplicate labels"
    out = {
        "tokenizer_json_sha256": hashlib.sha256(open(tok_path, "rb").read()).hexdigest(),
        "tokenizer_json_bytes": os.path.getsize(tok_path),
        "add_special_tokens": False,
        "n_cases": len(CASES),
        "cases": [],
    }
    for label, text in CASES:
        enc = tok.encode(text, add_special_tokens=False)
        rt = tok.decode(enc.ids, skip_special_tokens=False)
        out["cases"].append({
            "label": label,
            "text_utf8_hex": text.encode("utf-8").hex(),
            "n_bytes": len(text.encode("utf-8")),
            "ids": enc.ids,
            "n_ids": len(enc.ids),
            "decode_roundtrip_equal": rt == text,
        })
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    bad = [c["label"] for c in out["cases"] if not c["decode_roundtrip_equal"]]
    print("wrote %s: %d cases, %d total ids, roundtrip failures: %s" % (OUT, len(CASES), sum(c["n_ids"] for c in out["cases"]), bad))
    for c in out["cases"]:
        print("  %-28s bytes=%3d ids=%3d %s" % (c["label"], c["n_bytes"], c["n_ids"], c["ids"][:12]))


if __name__ == "__main__":
    main()
