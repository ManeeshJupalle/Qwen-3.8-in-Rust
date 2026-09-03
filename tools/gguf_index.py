"""Phase 0 GGUF / safetensors index capture. Read-only; never copies tensor data.

Outputs
  tests/fixtures/gguf_metadata.json  every KV metadata pair of the GGUF
  tests/fixtures/gguf_index.json     every tensor: name, ggml type, shape, byte offset, byte size
  tests/fixtures/st_index.json       every safetensors tensor: name, dtype, shape, shard, offsets
                                     (shapes come from shard headers fetched by tools/st_headers.py)
  docs/data/gguf_layout.txt          layout analysis (per-layer contiguity, largest layer, non-layer
                                     tensors, vision tower / tied-embed / MTP presence, HF-vs-GGUF diff)

Usage
  python tools/gguf_index.py <model.gguf> <hf_local_dir> [--tag NAME] [--no-st]
  --tag writes to tests/fixtures/gguf_metadata.<NAME>.json etc. (used for the MTP GGUF)
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np
from gguf import GGUFReader, GGUFValueType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
DOCS = os.path.join(ROOT, "docs", "data")

LAYER_RE = re.compile(r"^blk\.(\d+)\.")
HF_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def field_value(f):
    """Decode a gguf ReaderField into a plain Python value (works across gguf-py versions)."""
    if hasattr(f, "contents"):
        try:
            v = f.contents()
            if isinstance(v, np.ndarray):
                v = v.tolist()
            return v
        except Exception:
            pass
    t = f.types
    if not t:
        return None
    if t[0] == GGUFValueType.ARRAY:
        et = t[1] if len(t) > 1 else None
        if et == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
        out = []
        for i in f.data:
            out.extend(np.asarray(f.parts[i]).tolist())
        return out
    if t[0] == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
    return np.asarray(f.parts[f.data[0]]).tolist()[0]


def dump_metadata(reader, path):
    meta = {}
    for name, f in reader.fields.items():
        v = field_value(f)
        entry = {"types": [t.name for t in f.types], "value": v}
        if isinstance(v, list):
            entry["len"] = len(v)
        meta[name] = entry
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    return meta


def tensor_records(reader):
    recs = []
    for t in reader.tensors:
        # gguf-py reports shape in GGUF order (ne[0] is the fastest-varying dim, i.e. reversed vs numpy)
        shape_gguf = [int(x) for x in t.shape.tolist()]
        # the last part of the tensor-info field is the offset relative to the data section
        rel = int(np.asarray(t.field.parts[-1]).tolist()[0])
        recs.append({
            "name": t.name,
            "ggml_type": t.tensor_type.name,
            "ggml_type_id": int(t.tensor_type),
            "shape_gguf_order": shape_gguf,
            "shape_numpy_order": list(reversed(shape_gguf)),
            "n_elements": int(t.n_elements),
            "rel_offset": rel,
            "byte_offset": int(t.data_offset),
            "byte_size": int(t.n_bytes),
            "byte_end": int(t.data_offset) + int(t.n_bytes),
        })
    return recs


def yn(b):
    return "yes" if b else "no"


def layout_report(reader, recs, gguf_path, out_lines):
    P = out_lines.append
    file_size = os.path.getsize(gguf_path)
    align = int(reader.alignment)
    P("file: " + os.path.basename(gguf_path))
    P("file size bytes: %d" % file_size)
    P("alignment: %d" % align)
    P("data section starts at byte: %d" % int(reader.data_offset))
    P("total tensors: %d" % len(recs))
    total_bytes = sum(r["byte_size"] for r in recs)
    P("total tensor bytes: %d  (%.3f GiB)" % (total_bytes, total_bytes / 2**30))
    P("header+index bytes (file - tensor bytes): %d" % (file_size - total_bytes))
    P("")
    P("bytes per ggml type:")
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        by_type[r["ggml_type"]][0] += 1
        by_type[r["ggml_type"]][1] += r["byte_size"]
    for k, (n, b) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        P("  %-10s tensors=%5d bytes=%14d (%8.3f GiB, %5.1f%%)" % (k, n, b, b / 2**30, 100 * b / total_bytes))
    P("")

    layers = collections.defaultdict(list)
    non_layer = []
    for r in recs:
        m = LAYER_RE.match(r["name"])
        if m:
            layers[int(m.group(1))].append(r)
        else:
            non_layer.append(r)
    by_offset = sorted(recs, key=lambda r: r["byte_offset"])
    order_index = {r["name"]: i for i, r in enumerate(by_offset)}

    if layers:
        P("layers found: %d (indices %d..%d, 0-based as written in tensor names)" % (len(layers), min(layers), max(layers)))
    else:
        P("layers found: 0 (no blk.N.* tensors)")
    P("")
    P("per-layer layout (offsets are absolute file byte offsets):")
    largest = (None, -1)
    all_contig = True
    any_contig = False
    padding_only_layers = 0
    sizes = {}
    for L in sorted(layers):
        ts = sorted(layers[L], key=lambda r: r["byte_offset"])
        lo = ts[0]["byte_offset"]
        hi = max(r["byte_end"] for r in ts)
        ssum = sum(r["byte_size"] for r in ts)
        sizes[L] = ssum
        span = hi - lo
        gap = span - ssum
        foreign = [r["name"] for r in by_offset if lo <= r["byte_offset"] < hi and r not in ts]
        idxs = sorted(order_index[r["name"]] for r in ts)
        adjacent = idxs[-1] - idxs[0] + 1 == len(idxs)
        gaps = [b["byte_offset"] - a["byte_end"] for a, b in zip(ts, ts[1:])]
        max_gap = max(gaps) if gaps else 0
        contiguous = adjacent and not foreign and max_gap < align
        if contiguous:
            any_contig = True
            if gap > 0:
                padding_only_layers += 1
        else:
            all_contig = False
        if span > largest[1]:
            largest = (L, span)
        P("layer %2d: tensors=%2d min_offset=%d max_end=%d sum_sizes=%d span=%d contiguous: %s "
          "(gap bytes=%d, max single gap=%d, adjacent in file order=%s, foreign tensors inside span=%d)"
          % (L, len(ts), lo, hi, ssum, span, yn(contiguous), gap, max_gap, yn(adjacent), len(foreign)))
        for r in ts:
            P("    %-40s %-8s shape=%s off=%d size=%d" % (r["name"], r["ggml_type"], r["shape_numpy_order"], r["byte_offset"], r["byte_size"]))
        if foreign:
            P("    FOREIGN inside span: %s%s" % (foreign[:8], " ..." if len(foreign) > 8 else ""))
    P("")
    if layers:
        summary = "yes" if all_contig else ("partial" if any_contig else "no")
        P("ALL LAYERS CONTIGUOUS: %s  (layers with padding-only gaps: %d)" % (summary, padding_only_layers))
        P("LARGEST LAYER (by span, sizes the ring slot): layer %d, %d bytes = %.2f MiB" % (largest[0], largest[1], largest[1] / 2**20))
        big = max(sizes, key=sizes.get)
        small = min(sizes, key=sizes.get)
        P("LARGEST LAYER (by sum of tensor sizes): layer %d, %d bytes = %.2f MiB" % (big, sizes[big], sizes[big] / 2**20))
        P("smallest layer by sum: layer %d, %d bytes = %.2f MiB" % (small, sizes[small], sizes[small] / 2**20))
        sigs = collections.Counter()
        sig_layers = collections.defaultdict(list)
        for L in sorted(layers):
            sig = tuple(sorted(LAYER_RE.sub("", r["name"]) for r in layers[L]))
            sigs[sig] += 1
            sig_layers[sig].append(L)
        P("")
        P("distinct layer signatures (set of tensor suffixes):")
        for sig, n in sigs.items():
            P("  %d layers %s:" % (n, sig_layers[sig]))
            for s in sig:
                P("      " + s)
    P("")
    P("non-layer tensors (embed, output, norms, MTP, anything else):")
    nl_total = 0
    for r in sorted(non_layer, key=lambda r: r["byte_offset"]):
        nl_total += r["byte_size"]
        P("  %-40s %-8s shape=%s off=%d size=%d (%.2f MiB)" % (r["name"], r["ggml_type"], r["shape_numpy_order"], r["byte_offset"], r["byte_size"], r["byte_size"] / 2**20))
    P("  non-layer total bytes: %d (%.2f MiB)" % (nl_total, nl_total / 2**20))
    P("")
    names = {r["name"] for r in recs}
    vision = sorted(n for n in names if re.search(r"^(v\.|mm\.|vision|visual|mmproj)", n))
    P("vision tower present in GGUF: %s  (matched names: %s)" % ("YES" if vision else "no", vision[:5]))
    has_output = "output.weight" in names
    has_embd = "token_embd.weight" in names
    P("output.weight present: %s; token_embd.weight present: %s" % (yn(has_output), yn(has_embd)))
    P("embeddings tied (no separate output.weight): %s" % ("YES" if (has_embd and not has_output) else "no"))
    mtp = sorted(n for n in names if re.search(r"mtp|nextn|draft|eh_proj|enorm|hnorm|shared_head", n, re.I))
    P("MTP tensors present in GGUF: %s  (%s)" % ("YES" if mtp else "NO", mtp if mtp else "none matched /mtp|nextn|draft|eh_proj|enorm|hnorm|shared_head/"))
    other = sorted(n for n in names if not LAYER_RE.match(n) and n not in ("token_embd.weight", "output.weight", "output_norm.weight"))
    P("other non-layer tensors not in (token_embd, output, output_norm): %s" % other)
    return layers, non_layer


def load_st_index(hf_dir):
    idx = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    hdrs = {}
    for p in sorted(glob.glob(os.path.join(hf_dir, "shard_headers", "*.header.json"))):
        shard = os.path.basename(p).replace(".header.json", "")
        h = json.load(open(p))["header"]
        for k, v in h.items():
            if k == "__metadata__":
                continue
            hdrs[k] = (shard, v)
    recs = []
    for name in sorted(wm):
        rec = {"name": name, "shard": wm[name]}
        if name in hdrs:
            shard, v = hdrs[name]
            n = 1
            for d in v["shape"]:
                n *= d
            rec.update({"dtype": v["dtype"], "shape": v["shape"], "n_elements": n,
                        "data_offsets": v["data_offsets"],
                        "byte_size": v["data_offsets"][1] - v["data_offsets"][0]})
            if shard != wm[name]:
                rec["shard_mismatch"] = shard
        else:
            rec["shape"] = None
            rec["n_elements"] = -1
        recs.append(rec)
    return idx.get("metadata"), recs


def diff_names(st_recs, gg_recs, out_lines):
    P = out_lines.append
    P("")
    P("=" * 78)
    P("HF safetensors vs GGUF tensor-name diff")
    P("=" * 78)
    gg_names = [r["name"] for r in gg_recs]
    st_text = [r for r in st_recs if r["name"].startswith("model.language_model.") or r["name"] == "lm_head.weight"]
    st_vis = [r for r in st_recs if r["name"].startswith("model.visual.")]
    st_mtp = [r for r in st_recs if r["name"].startswith("mtp.")]
    known = set(r["name"] for r in st_text + st_vis + st_mtp)
    st_other = [r for r in st_recs if r["name"] not in known]
    P("HF tensors: %d = text %d + vision %d + mtp %d + other %d" % (len(st_recs), len(st_text), len(st_vis), len(st_mtp), len(st_other)))
    P("GGUF tensors: %d" % len(gg_recs))
    P("HF 'other' (neither text/vision/mtp): %s" % [r["name"] for r in st_other])
    P("")
    P("HF tensor families with NO GGUF counterpart:")
    gg_has_vision = any(n.startswith(("v.", "mm.")) for n in gg_names)
    gg_has_mtp = any(re.search(r"mtp|nextn", n) for n in gg_names)
    P("  vision tower: %d tensors (model.visual.*) -> %s" % (len(st_vis), "GGUF has v./mm. tensors" if gg_has_vision else "none in GGUF"))
    P("  mtp head:     %d tensors (mtp.*) -> %s" % (len(st_mtp), "present in GGUF" if gg_has_mtp else "none in GGUF"))
    for r in st_mtp:
        P("      %-55s %s %s" % (r["name"], r.get("dtype"), r.get("shape")))
    P("")
    P("per-layer element-count match (HF text layer L vs GGUF blk.L); an HF tensor is 'matched' if a GGUF tensor")
    P("in the same layer has the identical element count (names are NOT assumed):")
    hf_layers = collections.defaultdict(list)
    for r in st_text:
        m = HF_LAYER_RE.match(r["name"])
        if m:
            hf_layers[int(m.group(1))].append(r)
    gg_layers = collections.defaultdict(list)
    for r in gg_recs:
        m = LAYER_RE.match(r["name"])
        if m:
            gg_layers[int(m.group(1))].append(r)
    unmatched_hf_all = collections.Counter()
    unmatched_gg_all = collections.Counter()
    pairing_examples = {}
    for L in sorted(set(hf_layers) | set(gg_layers)):
        hf = sorted(hf_layers.get(L, []), key=lambda r: r["name"])
        gg = sorted(gg_layers.get(L, []), key=lambda r: r["name"])
        gg_pool = collections.Counter(r["n_elements"] for r in gg)
        un_hf = []
        for r in hf:
            if gg_pool[r["n_elements"]] > 0:
                gg_pool[r["n_elements"]] -= 1
            else:
                un_hf.append(r)
        hf_pool = collections.Counter(r["n_elements"] for r in hf)
        un_gg = []
        for r in gg:
            if hf_pool[r["n_elements"]] > 0:
                hf_pool[r["n_elements"]] -= 1
            else:
                un_gg.append(r)
        if L in (0, 3):
            pairing_examples[L] = (hf, gg)
        for r in un_hf:
            unmatched_hf_all[HF_LAYER_RE.sub("layers.N.", r["name"])] += 1
        for r in un_gg:
            unmatched_gg_all[LAYER_RE.sub("blk.N.", r["name"])] += 1
        if un_hf or un_gg:
            P("  layer %d: HF-unmatched=%s GGUF-unmatched=%s" % (
                L,
                [HF_LAYER_RE.sub("", r["name"]) + str(r["shape"]) for r in un_hf],
                [LAYER_RE.sub("", r["name"]) + str(r["shape_numpy_order"]) for r in un_gg]))
    if not unmatched_hf_all and not unmatched_gg_all:
        P("  every HF text-layer tensor has a same-size GGUF tensor in the same layer, and vice versa (all layers)")
    else:
        P("  HF text-layer tensors without a same-size GGUF tensor (pattern: count): %s" % dict(unmatched_hf_all))
        P("  GGUF blk tensors without a same-size HF tensor (pattern: count): %s" % dict(unmatched_gg_all))
    for L, (hf, gg) in pairing_examples.items():
        P("")
        P("  side-by-side for layer %d (sorted by element count then name; size-based pairing, NOT a verified name map):" % L)
        hf_s = sorted(hf, key=lambda r: (r["n_elements"], r["name"]))
        gg_s = sorted(gg, key=lambda r: (r["n_elements"], r["name"]))
        for i in range(max(len(hf_s), len(gg_s))):
            a = hf_s[i] if i < len(hf_s) else None
            b = gg_s[i] if i < len(gg_s) else None
            la = "%s %s n=%d" % (HF_LAYER_RE.sub("", a["name"]), a["shape"], a["n_elements"]) if a else "-"
            lb = "%s %s %s n=%d" % (LAYER_RE.sub("", b["name"]), b["shape_numpy_order"], b["ggml_type"], b["n_elements"]) if b else "-"
            P("    %-62s | %s" % (la, lb))
    P("")
    P("non-layer HF text tensors vs GGUF non-layer tensors (by element count):")
    hf_nl = [r for r in st_text if not HF_LAYER_RE.match(r["name"])]
    gg_nl = [r for r in gg_recs if not LAYER_RE.match(r["name"])]
    for r in hf_nl:
        same = [g["name"] for g in gg_nl if g["n_elements"] == r["n_elements"]]
        P("  %-45s %s n=%d -> GGUF same-size: %s" % (r["name"], r["shape"], r["n_elements"], same))
    for g in gg_nl:
        same = [r["name"] for r in hf_nl if r["n_elements"] == g["n_elements"]]
        if not same:
            P("  GGUF %s %s n=%d -> no same-size HF text tensor" % (g["name"], g["shape_numpy_order"], g["n_elements"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("hf_dir")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-st", action="store_true", help="skip the safetensors part")
    a = ap.parse_args()
    suffix = ("." + a.tag) if a.tag else ""
    os.makedirs(FIX, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)

    reader = GGUFReader(a.gguf, "r")
    dump_metadata(reader, os.path.join(FIX, "gguf_metadata%s.json" % suffix))
    recs = tensor_records(reader)
    with open(os.path.join(FIX, "gguf_index%s.json" % suffix), "w", encoding="utf-8") as fh:
        json.dump({"file": os.path.basename(a.gguf), "file_size": os.path.getsize(a.gguf),
                   "alignment": int(reader.alignment), "data_offset": int(reader.data_offset),
                   "n_tensors": len(recs), "tensors": recs}, fh, indent=1)
    lines = ["# generated by tools/gguf_index.py; read-only scan of the GGUF header + safetensors shard headers", ""]
    layout_report(reader, recs, a.gguf, lines)
    if not a.no_st:
        st_meta, st_recs = load_st_index(a.hf_dir)
        with open(os.path.join(FIX, "st_index.json"), "w", encoding="utf-8") as fh:
            json.dump({"metadata": st_meta, "n_tensors": len(st_recs), "tensors": st_recs}, fh, indent=1)
        diff_names(st_recs, recs, lines)
    out = os.path.join(DOCS, "gguf_layout%s.txt" % suffix)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
