"""Compare the MTP block (blk.64.*) of a local GGUF against another GGUF's blk.64: names, shapes, ggml types,
and (for a chosen subset) the dequantized values, so we can say whether they are the same weights.

The "other" side is either a local file (--other-local) or a remote Hub file whose header was already
captured by tools/gguf_header_probe.py (--other-remote REPO PATH INDEX_JSON); in the remote case the tensor
bytes are fetched with HTTP Range requests (tens of MB), never the whole file.

Value test: all F32 tensors are compared bit-exact (identical source weights => identical bytes regardless of
the quantisation recipe). Quantised tensors are dequantized on both sides and compared by cosine similarity,
relative RMS error and max|diff|; two quantisations of the same bf16 tensor agree to ~1e-2 relative, unrelated
weights give cosine ~0.

Usage:
  python tools/mtp_blk64_diff.py models/.../main.gguf --other-remote unsloth/Qwen3.8-27B-GGUF \
      MTP/mtp-Qwen3.8-27B-Q4_0.gguf tests/fixtures/ud/gguf_index.mtp.json --out docs/data/mtp_blk64_diff.txt
  python tools/mtp_blk64_diff.py a.gguf --other-local b.gguf --out ...
  --tensors NAME,NAME   quantised tensors to value-compare (default: attn_k, attn_v, nextn.eh_proj); --all for every one
"""
import argparse
import json
import os
import sys

import numpy as np
import requests
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize, quant_shape_to_byte_shape
from huggingface_hub import hf_hub_url

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_index as GI  # noqa: E402

DEFAULT_TENSORS = ["blk.64.attn_k.weight", "blk.64.attn_v.weight", "blk.64.nextn.eh_proj.weight"]


class LocalSource:
    def __init__(self, path):
        self.path = path
        self.reader = GGUFReader(path, "r")
        self.recs = {r["name"]: r for r in GI.tensor_records(self.reader) if r["name"].startswith("blk.64.")}
        self.label = os.path.basename(path)

    def get_bytes(self, name):
        r = self.recs[name]
        with open(self.path, "rb") as fh:
            fh.seek(r["byte_offset"])
            b = fh.read(r["byte_size"])
        assert len(b) == r["byte_size"]
        return b


class RemoteSource:
    def __init__(self, repo, path, index_json):
        d = json.load(open(index_json))
        self.recs = {r["name"]: r for r in d["tensors"] if r["name"].startswith("blk.64.")}
        self.url = hf_hub_url(repo, path)
        self.label = repo + "/" + path
        self.cache = os.path.join(GI.ROOT, "models", "_partial", os.path.basename(path))
        os.makedirs(self.cache, exist_ok=True)
        self.fetched = 0

    def get_bytes(self, name):
        r = self.recs[name]
        local = os.path.join(self.cache, name + ".bin")
        if os.path.exists(local) and os.path.getsize(local) == r["byte_size"]:
            return open(local, "rb").read()
        headers = {"Range": "bytes=%d-%d" % (r["byte_offset"], r["byte_offset"] + r["byte_size"] - 1)}
        tok = os.environ.get("HF_TOKEN")
        if tok:
            headers["Authorization"] = "Bearer " + tok
        resp = requests.get(self.url, headers=headers, allow_redirects=True, timeout=600)
        resp.raise_for_status()
        assert resp.status_code == 206, resp.status_code
        b = resp.content
        assert len(b) == r["byte_size"], (len(b), r["byte_size"])
        open(local, "wb").write(b)
        self.fetched += len(b)
        return b


def deq(rec, raw):
    qtype = GGMLQuantizationType(rec["ggml_type_id"])
    shape = tuple(rec["shape_numpy_order"])
    arr = np.frombuffer(raw, dtype=np.uint8)
    if qtype == GGMLQuantizationType.F32:
        return arr.view(np.float32).reshape(shape)
    if qtype == GGMLQuantizationType.F16:
        return arr.view(np.float16).astype(np.float32).reshape(shape)
    bs = quant_shape_to_byte_shape(shape, qtype)
    return dequantize(arr.reshape(bs), qtype).astype(np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_gguf")
    ap.add_argument("--other-local")
    ap.add_argument("--other-remote", nargs=3, metavar=("REPO", "PATH", "INDEX_JSON"))
    ap.add_argument("--tensors", default=",".join(DEFAULT_TENSORS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = LocalSource(a.main_gguf)
    B = LocalSource(a.other_local) if a.other_local else RemoteSource(*a.other_remote)
    L = []
    P = L.append
    P("# blk.64 (MTP block) comparison: tools/mtp_blk64_diff.py")
    P("A = %s" % A.label)
    P("B = %s" % B.label)
    P("")
    names = sorted(set(A.recs) | set(B.recs))
    P("== names / shapes / types (A vs B)")
    P("%-40s %-22s %-22s %-8s %-8s %s" % ("tensor", "shape A", "shape B", "type A", "type B", "same shape"))
    same_names = set(A.recs) == set(B.recs)
    all_same_shape = True
    for n in names:
        ra, rb = A.recs.get(n), B.recs.get(n)
        sa = str(ra["shape_numpy_order"]) if ra else "-"
        sb = str(rb["shape_numpy_order"]) if rb else "-"
        ta = ra["ggml_type"] if ra else "-"
        tb = rb["ggml_type"] if rb else "-"
        same = bool(ra and rb and ra["shape_numpy_order"] == rb["shape_numpy_order"])
        all_same_shape &= same
        P("%-40s %-22s %-22s %-8s %-8s %s" % (n, sa, sb, ta, tb, "yes" if same else "NO"))
    P("")
    P("name sets identical: %s (A has %d, B has %d); all shapes identical: %s" % (same_names, len(A.recs), len(B.recs), all_same_shape))
    P("types identical: %s" % all(A.recs[n]["ggml_type"] == B.recs[n]["ggml_type"] for n in names if n in A.recs and n in B.recs))
    P("")

    P("== values: F32 tensors (bit-exact test)")
    f32_all_equal = True
    for n in names:
        if n in A.recs and n in B.recs and A.recs[n]["ggml_type"] == "F32" and B.recs[n]["ggml_type"] == "F32":
            xa = deq(A.recs[n], A.get_bytes(n))
            xb = deq(B.recs[n], B.get_bytes(n))
            eq = np.array_equal(xa, xb)
            f32_all_equal &= eq
            P("%-40s bit-identical: %-5s max|diff|=%.3e  A[:4]=%s" % (n, eq, float(np.abs(xa - xb).max()), np.round(xa.ravel()[:4], 5).tolist()))
    P("")

    P("== values: quantised tensors (dequantized on both sides)")
    sel = [n for n in names if n in A.recs and n in B.recs and A.recs[n]["ggml_type"] != "F32"]
    if not a.all:
        want = set(a.tensors.split(","))
        sel = [n for n in sel if n in want]
    cos_all = []
    for n in sel:
        xa = deq(A.recs[n], A.get_bytes(n)).ravel().astype(np.float64)
        xb = deq(B.recs[n], B.get_bytes(n)).ravel().astype(np.float64)
        cos = float(np.dot(xa, xb) / (np.linalg.norm(xa) * np.linalg.norm(xb) + 1e-30))
        rel_rms = float(np.sqrt(np.mean((xa - xb) ** 2)) / (np.sqrt(np.mean(xa ** 2)) + 1e-30))
        cos_all.append(cos)
        P("%-40s A=%-6s B=%-6s cosine=%.6f rel_rms_err=%.4f max|diff|=%.4f rms(A)=%.4f rms(B)=%.4f" % (
            n, A.recs[n]["ggml_type"], B.recs[n]["ggml_type"], cos, rel_rms, float(np.abs(xa - xb).max()),
            float(np.sqrt(np.mean(xa ** 2))), float(np.sqrt(np.mean(xb ** 2)))))
    P("")
    if isinstance(B, RemoteSource):
        P("bytes fetched from B by range request: %d" % B.fetched)
    verdict = "SAME underlying weights" if (f32_all_equal and cos_all and min(cos_all) > 0.99) else \
              ("DIFFERENT weights" if (cos_all and max(cos_all) < 0.9) else "INCONCLUSIVE")
    P("VERDICT: %s (F32 tensors bit-identical: %s; min cosine over %d quantised tensors: %s)" % (
        verdict, f32_all_equal, len(cos_all), ("%.6f" % min(cos_all)) if cos_all else "n/a"))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
