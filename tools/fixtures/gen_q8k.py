"""Q8_K activation quantiser fixture (Phase 3): a Python port of ggml's `quantize_row_q8_K_ref`
(ggml/src/ggml-quants.c), since gguf-py has no Q8_K. The Rust `Q8KRow::quantize_scalar` and the AVX2 version
must reproduce every byte (tests/q8k.rs).

Reference semantics, per 256-value block:
  max   = the signed element with the largest |x| (first on ties: strict `>` scan == np.argmax of |x|)
  all-zero block -> d = 0, codes 0, bsums 0
  iscale = float32(-127) / max            (float32 divide)
  q      = min(127, nearest_int(iscale * x))   nearest_int = round half to even (np.rint) for |v| < 2^22
  bsums[g] = sum(q[16g:16g+16])            (int16)
  d      = float32(1) / iscale             (float32 divide; carries the sign of -max)
Block bytes: f32 d, 256 x int8, 16 x int16 (292 bytes), little-endian.
"""
import struct

import numpy as np
import torch

from common import Fixture

DEN = 1e-40


def quantize_q8k_row(row):
    """One row (width a multiple of 256) -> ggml block_q8_K bytes."""
    row = np.asarray(row, dtype=np.float32)
    assert row.size % 256 == 0
    out = bytearray()
    for b in range(row.size // 256):
        blk = row[b * 256:(b + 1) * 256]
        a = np.abs(blk)
        i = int(np.argmax(a))  # first occurrence of the maximum
        amax = a[i]
        if amax == 0:
            out += struct.pack("<f", 0.0) + bytes(256) + bytes(32)
            continue
        mx = blk[i]
        iscale = np.float32(np.float32(-127.0) / mx)
        prod = (iscale * blk).astype(np.float32)
        v = np.rint(prod).astype(np.int64)  # ties to even, like nearest_int
        q = np.minimum(127, v).astype(np.int8)
        bsums = q.reshape(16, 16).astype(np.int32).sum(axis=1).astype("<i2")
        d = np.float32(np.float32(1.0) / iscale)
        out += struct.pack("<f", float(d)) + q.astype("<i1").tobytes() + bsums.tobytes()
    return bytes(out)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, dtype=torch.float32) * scale


def gen_q8k():
    torch.manual_seed(20260903)
    fx = Fixture("q8k_quantize")
    fx.meta["source"] = "tools/fixtures/gen_q8k.py: port of ggml quantize_row_q8_K_ref (iscale = -127/max, nearest-even, min 127, bsums per 16, d = 1/iscale)"
    fx.meta["exact"] = True
    fx.meta["block_bytes"] = 292
    for width in (256, 512, 5120):
        ties = torch.tensor([(i % 9) - 4 + 0.5 for i in range(width)], dtype=torch.float32)
        ties[0] = -127.0  # iscale = 1 exactly, so every k + 0.5 is a tie
        ties[300 % width] = 127.0
        pm_pos = rnd(width, scale=0.5)
        pm_pos[7] = 4.0
        pm_pos[9] = -4.0
        pm_neg = pm_pos.clone()
        pm_neg[7] = -4.0
        pm_neg[9] = 4.0
        zero_block = rnd(width)
        zero_block[:256] = 0.0
        rows = [
            ("random", rnd(width)),
            ("large", rnd(width, scale=1e30)),
            ("tiny", rnd(width, scale=1e-30)),
            # every block gets one normal value: an all-denormal block overflows iscale to inf in ggml too
            # (the int conversion of inf is undefined there), so it is outside the contract, as for Q8_0
            ("denormal_mixed", torch.cat([torch.tensor([DEN, -DEN, 1.0]), torch.full((width - 3,), DEN)]).index_fill_(0, torch.arange(0, width, 256), 1.0)),
            ("neg_zero", torch.full((width,), -0.0)),
            ("all_equal", torch.full((width,), 3.0)),
            ("max_last_negative", torch.cat([rnd(width - 1, scale=0.1), torch.tensor([-5.0])])),
            ("ties_even", ties),
            ("first_max_positive", pm_pos),
            ("first_max_negative", pm_neg),
            ("zero_block", zero_block),
            ("alternating", torch.tensor([5.0 if i % 2 == 0 else -5.0 for i in range(width)])),
        ]
        names, rs = zip(*rows)
        x = torch.stack(rs).numpy().astype(np.float32)
        raw = b"".join(quantize_q8k_row(r) for r in x)
        c = "q8k_w%d" % width
        fx.add(c, kind="q8k_quantize", width=width, rows=list(names), x=fx.array(c, "x", x), raw=fx.raw(c, "raw", raw))
    fx.write()


if __name__ == "__main__":
    gen_q8k()
