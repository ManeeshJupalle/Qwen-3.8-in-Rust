"""Per-ggml-type byte breakdown of one or more GGUF files, from headers only.

Sources:
  --remote REPO PATH   fetch the first --bytes of a Hub file with an HTTP Range request and parse the header
                       (repeatable; for split files pass every split and they are aggregated)
  --index-json FILE    an existing tests/fixtures/gguf_index*.json written by tools/gguf_index.py

Writes a text report to --out (default docs/data/gguf_type_breakdown.txt) and prints it.

Usage:
  python tools/gguf_type_breakdown.py --index-json tests/fixtures/ud/gguf_index.json \
      --remote bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --out docs/data/q4km_type_breakdown.txt
"""
import argparse
import collections
import json
import os
import re
import sys

import requests
from huggingface_hub import hf_hub_url

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_index as GI  # noqa: E402
from gguf_header_probe import HeaderOnlyReader  # noqa: E402

LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def fetch_prefix(repo_id, path, n_bytes, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(path) + ".head")
    if os.path.exists(local) and os.path.getsize(local) >= n_bytes:
        total = None
    else:
        headers = {"Range": "bytes=0-%d" % (n_bytes - 1)}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        r = requests.get(hf_hub_url(repo_id, path), headers=headers, allow_redirects=True, timeout=300)
        r.raise_for_status()
        assert r.status_code == 206, r.status_code
        total = int(r.headers.get("Content-Range", "bytes 0-0/0").split("/")[-1])
        with open(local, "wb") as fh:
            fh.write(r.content)
    return local, total


def breakdown(recs):
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by_type[r["ggml_type"]][0] += 1
        by_type[r["ggml_type"]][1] += r["byte_size"]
    return by_type


def report(label, recs, extra, L):
    P = L.append
    total = sum(r["byte_size"] for r in recs)
    bt = breakdown(recs)
    P("== %s" % label)
    for k, v in extra.items():
        P("   %s: %s" % (k, v))
    P("   tensors: %d   tensor bytes: %d (%.3f GiB)   distinct ggml types: %d" % (len(recs), total, total / 2**30, len(bt)))
    P("   %-8s %7s %15s %7s" % ("type", "tensors", "bytes", "share"))
    for k, (n, b) in sorted(bt.items(), key=lambda kv: -kv[1][1]):
        P("   %-8s %7d %15d %6.1f%%" % (k, n, b, 100.0 * b / total))
    # per-tensor-role type sets (which roles use which types)
    roles = collections.defaultdict(collections.Counter)
    for r in recs:
        role = LAYER_RE.sub("blk.N.", r["name"])
        roles[role][r["ggml_type"]] += 1
    P("   types by tensor role:")
    for role in sorted(roles):
        P("     %-36s %s" % (role, dict(roles[role])))
    names = {r["name"] for r in recs}
    layers = sorted({int(m.group(1)) for n in names for m in [LAYER_RE.match(n)] if m})
    P("   blocks: %d (%s..%s); has blk.64 (MTP): %s; nextn tensors: %d" % (
        len(layers), layers[0] if layers else "-", layers[-1] if layers else "-",
        "yes" if 64 in layers else "no", sum(1 for n in names if ".nextn." in n)))
    P("")
    return len(bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", nargs=2, action="append", default=[], metavar=("REPO", "PATH"))
    ap.add_argument("--index-json", action="append", default=[])
    ap.add_argument("--bytes", type=int, default=16 * 2**20)
    ap.add_argument("--out", default=os.path.join(GI.DOCS, "gguf_type_breakdown.txt"))
    a = ap.parse_args()
    L = ["# ggml-type breakdown from GGUF headers only (tools/gguf_type_breakdown.py); no tensor data read", ""]
    results = []
    for path in a.index_json:
        d = json.load(open(path))
        n = report("%s (local index %s)" % (d["file"], path), d["tensors"], {"file_size": d["file_size"]}, L)
        results.append((d["file"], n))
    for repo, path in a.remote:
        local, total = fetch_prefix(repo, path, a.bytes, os.path.join(GI.ROOT, "models", "_headers"))
        reader = HeaderOnlyReader(local, "r")
        assert int(reader.data_offset) <= os.path.getsize(local), "header longer than fetched prefix; raise --bytes"
        recs = GI.tensor_records(reader)
        arch = GI.field_value(reader.fields["general.architecture"])
        ft = GI.field_value(reader.fields["general.file_type"]) if "general.file_type" in reader.fields else None
        extra = {"repo": repo, "remote_file_size": total if total is not None else "(cached header)",
                 "general.architecture": arch, "general.file_type": ft,
                 "header_bytes (data_offset)": int(reader.data_offset), "block_count": GI.field_value(reader.fields[arch + ".block_count"]) if (arch + ".block_count") in reader.fields else None}
        n = report("%s/%s (remote header)" % (repo, path), recs, extra, L)
        results.append((repo + "/" + path, n))
    L.append("== summary: distinct ggml types per file")
    for name, n in results:
        L.append("   %2d  %s" % (n, name))
    if results:
        best = min(results, key=lambda x: x[1])
        L.append("   fewest types: %s (%d)" % (best[0], best[1]))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
