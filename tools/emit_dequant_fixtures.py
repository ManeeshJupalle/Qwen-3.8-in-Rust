"""Emit dequantisation parity fixtures for every ggml type in the primary GGUF, using gguf-py as the spec.

For each type present in the file (F32, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K):
  full    : the SMALLEST tensor of that type, dequantised with gguf.quants.dequantize.
            raw bytes are stored when <= 8 MiB (else the test reads them from the GGUF by offset);
            the f32 output is stored as the first 4096 values (.npy) plus CRC-32 of the full f32 LE byte
            stream (zlib.crc32), float64 sum, max|x|, so full-tensor equality is checked without a 100 MB fixture.
  prefix  : the first 3 blocks of the LARGEST tensor of that type: raw bytes + full f32 .npy.
  hostile : hand-built single blocks whose scale/min/high-bit fields are all distinct, so a wrong bit-unpack
            fails visibly: raw bytes + f32 .npy (dequantised by gguf-py).
Everything is listed in tests/fixtures/dequant/manifest.json.

Usage: python tools/emit_dequant_fixtures.py <model.gguf>
"""
import json
import os
import struct
import sys
import zlib

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize, quant_shape_to_byte_shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "dequant")
RAW_LIMIT = 8 * 2**20
HEAD = 4096
T = GGMLQuantizationType


def f16(x):
    return struct.pack("<e", x)


def deq_bytes(raw, qtype, shape):
    arr = np.frombuffer(raw, dtype=np.uint8)
    if qtype == T.F32:
        return arr.view(np.float32).reshape(shape)
    bs = quant_shape_to_byte_shape(shape, qtype)
    return dequantize(arr.reshape(bs), qtype).astype(np.float32).reshape(shape)


def digest(f32):
    b = np.ascontiguousarray(f32, dtype="<f4").tobytes()
    return {"crc32": zlib.crc32(b) & 0xFFFFFFFF, "sum_f64": float(np.sum(f32, dtype=np.float64)),
            "max_abs": float(np.max(np.abs(f32))), "n": int(f32.size)}


def pack_scales_k4(sc, mn):
    """Inverse of get_scale_min_k4 (ggml quantize_row_q4_K_ref packing)."""
    s = bytearray(12)
    for j in range(8):
        if j < 4:
            s[j] = sc[j]
            s[j + 4] = mn[j]
        else:
            s[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4)
            s[j - 4] |= (sc[j] >> 4) << 6
            s[j] |= (mn[j] >> 4) << 6
    return bytes(s)


def unpack_scales_k4(s):
    out = []
    for j in range(8):
        if j < 4:
            out.append((s[j] & 63, s[j + 4] & 63))
        else:
            out.append(((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)))
    return out


def hostile_blocks():
    """Hand-built blocks. Returns list of (label, qtype, raw_bytes, description)."""
    out = []
    # Q4_0: d = 1.5; nibbles cover 0..15 in both halves in different orders
    qs = bytes(((i * 7 + 3) & 0xF) | ((((i * 11) + 5) & 0xF) << 4) for i in range(16))
    out.append(("q4_0", T.Q4_0, f16(1.5) + qs, "d=1.5, low/high nibbles distinct permutations"))
    # Q8_0: d = 0.75; int8 pattern including -128 and 127
    q8 = bytes((i * 37 - 128) & 0xFF for i in range(32))
    q8 = bytes([0x80, 0x7F]) + q8[2:]
    out.append(("q8_0", T.Q8_0, f16(0.75) + q8, "d=0.75, int8 values incl -128, 127"))
    # Q4_K: distinct 6-bit scales and mins using both halves of every packed byte
    sc = [63, 1, 2, 4, 8, 16, 32, 33]
    mn = [62, 3, 5, 9, 17, 31, 47, 60]
    scales = pack_scales_k4(sc, mn)
    assert unpack_scales_k4(scales) == list(zip(sc, mn))
    qs = bytes(((i * 5 + 1) & 0xF) | ((((i * 3) + 7) & 0xF) << 4) for i in range(128))
    out.append(("q4_k", T.Q4_K, f16(1.5) + f16(0.25) + scales + qs,
                "d=1.5 dmin=0.25 sc=%s m=%s, nibbles distinct" % (sc, mn)))
    # Q5_K: same scales, qh plane with every bit position varying per byte
    qh = bytes((i * 37 + 11) & 0xFF for i in range(32))
    ql = bytes(((i * 9 + 2) & 0xF) | ((((i * 13) + 4) & 0xF) << 4) for i in range(128))
    out.append(("q5_k", T.Q5_K, f16(1.5) + f16(0.25) + scales + qh + ql,
                "d=1.5 dmin=0.25 same scales; qh bits vary per byte; ql distinct"))
    # Q6_K: ql/qh patterns, 16 distinct int8 scales incl negatives and extremes, d at the end
    ql = bytes(((i * 7 + 1) & 0xF) | ((((i * 5) + 3) & 0xF) << 4) for i in range(128))
    qh = bytes((i * 29 + 6) & 0xFF for i in range(64))
    sc6 = [127, -128, 1, -1, 2, -3, 5, -8, 13, -21, 34, -55, 89, -100, 64, -64]
    scales6 = bytes(s & 0xFF for s in sc6)
    out.append(("q6_k", T.Q6_K, ql + qh + scales6 + f16(0.5),
                "d=0.5, int8 scales %s, ql/qh distinct patterns" % sc6))
    # F32: a few special values
    f = np.array([0.0, -0.0, 1.0, -1.5, 3.4e38, 1e-45, float("inf"), -float("inf")], dtype="<f4")
    out.append(("f32", T.F32, f.tobytes(), "special float32 values incl inf and subnormal"))
    return out


def main():
    gguf_path = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    reader = GGUFReader(gguf_path, "r")
    manifest = {"gguf": os.path.basename(gguf_path), "gguf_bytes": os.path.getsize(gguf_path),
                "spec": "gguf-py quants.dequantize (gguf package %s)" % __import__("importlib.metadata").metadata.version("gguf"),
                "digest": "crc32 = zlib.crc32 over the float32 little-endian byte stream",
                "head_values": HEAD, "raw_limit_bytes": RAW_LIMIT, "types": {}, "hostile": []}
    by_type = {}
    for t in reader.tensors:
        by_type.setdefault(t.tensor_type, []).append(t)
    for qtype in sorted(by_type, key=lambda q: q.name):
        ts = by_type[qtype]
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        smallest = min(ts, key=lambda t: (t.n_bytes, t.name))
        largest = max(ts, key=lambda t: (t.n_bytes, t.name))
        entry = {"ggml_type": qtype.name, "ggml_type_id": int(qtype), "block_size": block_size, "type_size": type_size,
                 "n_tensors": len(ts)}
        # full: smallest tensor
        shape = tuple(reversed(smallest.shape.tolist()))
        raw = np.asarray(smallest.data).tobytes()
        assert len(raw) == smallest.n_bytes, (len(raw), smallest.n_bytes)
        f32 = deq_bytes(raw, qtype, shape)
        head = np.ascontiguousarray(f32.ravel()[:HEAD], dtype="<f4")
        np.save(os.path.join(OUT, "%s_full_head.npy" % qtype.name), head)
        full = {"tensor": smallest.name, "shape_numpy": list(shape), "n_elements": int(smallest.n_elements),
                "byte_size": int(smallest.n_bytes), "abs_offset": int(smallest.data_offset),
                "head_npy": "%s_full_head.npy" % qtype.name, "digest": digest(f32.ravel())}
        if smallest.n_bytes <= RAW_LIMIT:
            open(os.path.join(OUT, "%s_full.bin" % qtype.name), "wb").write(raw)
            full["raw_bin"] = "%s_full.bin" % qtype.name
        else:
            full["raw_bin"] = None
            full["raw_note"] = "raw bytes exceed %d; the test reads them from the GGUF at abs_offset" % RAW_LIMIT
        entry["full"] = full
        # prefix: first 3 blocks of the largest tensor
        nblk = 3
        praw = np.asarray(largest.data).tobytes()[: nblk * type_size]
        pf32 = deq_bytes(praw, qtype, (nblk * block_size,))
        open(os.path.join(OUT, "%s_prefix.bin" % qtype.name), "wb").write(praw)
        np.save(os.path.join(OUT, "%s_prefix.npy" % qtype.name), np.ascontiguousarray(pf32.ravel(), dtype="<f4"))
        entry["prefix"] = {"tensor": largest.name, "n_blocks": nblk, "n_elements": nblk * block_size,
                           "raw_bin": "%s_prefix.bin" % qtype.name, "npy": "%s_prefix.npy" % qtype.name,
                           "abs_offset": int(largest.data_offset)}
        manifest["types"][qtype.name] = entry
        print("%-6s full=%s (%d B, %d elems) prefix=%s" % (qtype.name, smallest.name, smallest.n_bytes, smallest.n_elements, largest.name))
    for label, qtype, raw, desc in hostile_blocks():
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        n = len(raw) // type_size * block_size
        f32 = deq_bytes(raw, qtype, (n,))
        open(os.path.join(OUT, "hostile_%s.bin" % label), "wb").write(raw)
        np.save(os.path.join(OUT, "hostile_%s.npy" % label), np.ascontiguousarray(f32.ravel(), dtype="<f4"))
        manifest["hostile"].append({"label": label, "ggml_type": qtype.name, "n_elements": n, "raw_bin": "hostile_%s.bin" % label,
                                    "npy": "hostile_%s.npy" % label, "description": desc,
                                    "values_head": [float(x) if np.isfinite(x) else str(x) for x in f32.ravel()[:8]]})
        print("hostile %-5s %s -> %s" % (label, desc[:50], np.round(f32.ravel()[:6], 4).tolist()))
    text = json.dumps(manifest, indent=1, allow_nan=False)  # inf/nan would not be valid JSON for the Rust side
    open(os.path.join(OUT, "manifest.json"), "w").write(text)
    print("wrote", os.path.join(OUT, "manifest.json"))


if __name__ == "__main__":
    main()
