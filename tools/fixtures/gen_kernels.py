"""Kernel fixtures for Phase 2a. Spec = the HF Qwen3.5 module functions (transformers 5.16.1) and gguf-py.

Each kernel gets tests/fixtures/kernels/<kernel>/manifest.json plus .npy/.bin files. Every case records
`max_abs` (largest magnitude the kernel touches) and the manifest records `budget_k`, so the Rust test
computes budget = k * sqrt(width) * f32_eps * max_abs (rule 5) from the fixture, never from a typed number.

Hostile inputs per kernel: denormals, large magnitude, negative zero, width not a multiple of 32 (where the
kernel allows it), all-equal inputs.

gguf-py can quantize only Q4_0 and Q8_0, so K-quant weight rows come from the real bartowski GGUF plus
hostile blocks tiled from tests/fixtures/dequant/hostile_*.bin (distinct 6-bit scales) with rewritten f16 scales.

Usage: python tools/fixtures/gen_kernels.py
"""
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Fixture  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

torch.manual_seed(20260903)
np.random.seed(20260903)
DEN = 1e-40  # float32 subnormal
EPS = 1e-6
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GGUF_PATH = os.environ.get("AQUEDUCT_GGUF", r"C:\models\Qwen3.8-27B-Q4_K_M.gguf")  # on the SSD since Phase 3 (finding 34)
DEQ_DIR = os.path.join(ROOT, "tests", "fixtures", "dequant")
_reader = None


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, dtype=torch.float32) * scale


def hostile_rows(width, base_scale=1.0):
    """Rows that stress a row-wise kernel: random, denormal, large, -0.0, all-equal, mixed."""
    return [
        ("random", rnd(width, scale=base_scale)),
        ("denormal", torch.full((width,), DEN, dtype=torch.float32) * torch.sign(rnd(width))),
        ("large", rnd(width, scale=1e18)),
        ("neg_zero", torch.full((width,), -0.0, dtype=torch.float32)),
        ("all_equal", torch.full((width,), 3.0, dtype=torch.float32)),
        ("mixed", torch.cat([torch.tensor([DEN, -DEN, 0.0, -0.0, 1e18, -1e18]), rnd(width - 6, scale=base_scale)])),
    ]


# --------------------------------------------------------------------------------------------- rmsnorm
def gen_rmsnorm():
    fx = Fixture("rmsnorm")
    fx.meta["source"] = "modeling_qwen3_5.Qwen3_5RMSNorm (lines 723-742): y = x * rsqrt(mean(x^2) + eps) * (1 + w); gated: Qwen3_5RMSNormGated (168-186)"
    fx.meta["budget_k"] = 8
    fx.meta["weight_note"] = "w_gguf = 1 + w_hf for the zero-centered norm (converter adds 1); gated norm weight stored as-is"
    for width in (64, 100, 256, 5120):
        norm = M.Qwen3_5RMSNorm(width, eps=EPS)
        with torch.no_grad():
            norm.weight.copy_(rnd(width, scale=0.1))
        names, rows = zip(*hostile_rows(width))
        x = torch.stack(rows)
        with torch.no_grad():
            y = norm(x)
        c = "rmsnorm_w%d" % width
        fx.add(c, kind="rmsnorm", width=width, eps=EPS, rows=list(names), x=fx.array(c, "x", x),
               w_gguf=fx.array(c, "w_gguf", norm.weight + 1.0), y=fx.array(c, "y", y), max_abs=float(y.abs().max()))
    for width in (8, 128):
        norm = M.Qwen3_5RMSNormGated(width, eps=EPS)
        with torch.no_grad():
            norm.weight.copy_(1.0 + rnd(width, scale=0.1))
        names, rows = zip(*hostile_rows(width))
        x = torch.stack(rows)
        gate = torch.stack([r for _, r in hostile_rows(width, base_scale=2.0)])
        with torch.no_grad():
            y = norm(x, gate)
        c = "rmsnorm_gated_w%d" % width
        fx.add(c, kind="rmsnorm_gated", width=width, eps=EPS, rows=list(names), x=fx.array(c, "x", x), gate=fx.array(c, "gate", gate),
               w_gguf=fx.array(c, "w_gguf", norm.weight), y=fx.array(c, "y", y), max_abs=float(y[torch.isfinite(y)].abs().max()))
    fx.write()


# --------------------------------------------------------------------------------------------- silu / swiglu
def gen_act():
    fx = Fixture("act")
    fx.meta["source"] = "torch.nn.functional.silu (ACT2FN['silu']); swiglu = silu(gate) * up, the elementwise part of Qwen3_5MLP (707-722)"
    fx.meta["budget_k"] = 8
    x = torch.cat([rnd(1000), torch.tensor([DEN, -DEN, 0.0, -0.0, 1e3, -1e3, 1e30, -1e30, 20.0, -20.0, 88.0, -88.0, 89.0, -89.0, 1e-3, -1e-3])])
    with torch.no_grad():
        y = F.silu(x)
    fx.add("silu", kind="silu", n=int(x.numel()), x=fx.array("silu", "x", x), y=fx.array("silu", "y", y), max_abs=float(y[torch.isfinite(y)].abs().max()))
    gate = torch.cat([rnd(1000), torch.tensor([DEN, -DEN, 0.0, -0.0, 1e3, -1e3, 50.0, -50.0])])
    up = torch.cat([rnd(1000), torch.tensor([1e3, 1e3, -1e3, 1e3, 1e-30, 1e30, 2.0, 2.0])])
    with torch.no_grad():
        y = F.silu(gate) * up
    fx.add("swiglu", kind="swiglu", n=int(gate.numel()), gate=fx.array("swiglu", "gate", gate), up=fx.array("swiglu", "up", up),
           y=fx.array("swiglu", "y", y), max_abs=float(y[torch.isfinite(y)].abs().max()))
    fx.write()


# --------------------------------------------------------------------------------------------- softmax
def gen_softmax():
    fx = Fixture("softmax")
    fx.meta["source"] = "torch.softmax(x, -1, dtype=float32) as in eager_attention_forward (line 618); causal: softmax(scores + mask) with mask = finfo(float32).min above the diagonal (create_causal_mask)"
    fx.meta["budget_k"] = 8
    for width in (5, 32, 100, 512):
        rows = [
            ("random", rnd(width, scale=3.0)),
            ("large", rnd(width, scale=1e30)),
            ("all_equal", torch.full((width,), 7.5)),
            ("outlier", torch.cat([torch.tensor([200.0]), rnd(width - 1)])),
            ("neg_inf", torch.cat([torch.tensor([float("-inf"), 1.0]), rnd(width - 2)])),
            ("denormal", torch.full((width,), DEN)),
            ("neg_zero", torch.full((width,), -0.0)),
        ]
        names, rs = zip(*rows)
        x = torch.stack(rs)
        with torch.no_grad():
            y = torch.softmax(x, dim=-1, dtype=torch.float32)
        c = "softmax_w%d" % width
        fx.add(c, kind="softmax", width=width, rows=list(names), x=fx.array(c, "x", x), y=fx.array(c, "y", y), max_abs=1.0)
    for tlen in (1, 7, 40):
        scores = rnd(tlen, tlen, scale=4.0)
        mask = torch.full((tlen, tlen), torch.finfo(torch.float32).min).triu(1)
        with torch.no_grad():
            y = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
        c = "causal_t%d" % tlen
        fx.add(c, kind="softmax_causal", t=tlen, scores=fx.array(c, "scores", scores), y=fx.array(c, "y", y), max_abs=1.0,
               mask_note="row i attends to columns 0..=i; masked columns must contribute exactly 0")
    fx.write()


# --------------------------------------------------------------------------------------------- q8 quantize
def gen_q8():
    fx = Fixture("q8_quantize")
    fx.meta["source"] = "gguf.quants.quantize(x, Q8_0) == ggml quantize_row_q8_0_ref: d = amax/127 (f32) -> f16; id = 1/d (f32, from the unrounded d); q = round-half-away(x*id)"
    fx.meta["exact"] = True
    fx.meta["note"] = "every 32-block gets one normal value: an all-denormal block is NOT generated: amax/127 underflows f16 to 0 while id overflows, and ggml's int8 conversion of inf is undefined; |x| > 127*65504 overflows the f16 scale likewise"
    for width in (32, 64, 5120):
        rows = [
            ("random", rnd(width)),
            ("denormal_mixed", torch.cat([torch.tensor([DEN, -DEN, 1.0]), torch.full((width - 3,), DEN)]).index_fill_(0, torch.arange(0, width, 32), 1.0)),
            ("large", rnd(width, scale=1e6)),
            ("neg_zero", torch.full((width,), -0.0)),
            ("all_equal", torch.full((width,), 3.0)),
            ("max_last", torch.cat([rnd(width - 1, scale=0.1), torch.tensor([-5.0])])),
            ("tie_pm", torch.cat([torch.tensor([4.0, -4.0]), rnd(width - 2, scale=0.5)])),
            ("half_round", torch.tensor([127.0] + [0.5 * (i % 7 - 3) for i in range(width - 1)])),
        ]
        names, rs = zip(*rows)
        x = torch.stack(rs).numpy().astype(np.float32)
        raw = quantize(x, T.Q8_0)
        deq = dequantize(raw, T.Q8_0).astype(np.float32)
        c = "q8_w%d" % width
        fx.add(c, kind="q8_quantize", width=width, rows=list(names), x=fx.array(c, "x", x), raw=fx.raw(c, "raw", raw.tobytes()),
               deq=fx.array(c, "deq", deq), block_bytes=34)
    fx.write()


# --------------------------------------------------------------------------------------------- dot kernels / matvec
QTYPES = [T.Q4_0, T.Q8_0, T.Q4_K, T.Q5_K, T.Q6_K, T.F32]


def real_rows(qtype, width, n_rows):
    """Rows of real weights of `qtype` from the bartowski GGUF (gguf-py has no K-quant quantizer)."""
    global _reader
    if _reader is None:
        _reader = GGUFReader(GGUF_PATH, "r")
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    for t in _reader.tensors:
        if t.tensor_type != qtype:
            continue
        row_elems = int(t.shape[0])  # GGUF ne[0] = fastest dim = row length
        if row_elems % width == 0 and int(t.shape[1]) >= n_rows:
            data = np.asarray(t.data)  # (rows, row_bytes) uint8
            want_bytes = width // block_size * type_size
            return t.name, [data[i, :want_bytes].tobytes() for i in range(n_rows)]
    raise RuntimeError("no %s tensor with rows divisible by %d" % (qtype.name, width))


def hostile_kquant_rows(qtype, width):
    """Rows tiled from the Phase 1 hostile block (distinct 6-bit scales) with the f16 scale fields rewritten:
    as-is, large scale (d = 60000), tiny scale (d = 6.1e-5), and an all-equal-values block."""
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    blk = bytearray(open(os.path.join(DEQ_DIR, "hostile_%s.bin" % qtype.name.lower()), "rb").read())
    assert len(blk) == type_size

    def with_scale(b, d, dmin=None):
        b = bytearray(b)
        if qtype == T.Q6_K:
            b[208:210] = struct.pack("<e", d)
        else:
            b[0:2] = struct.pack("<e", d)
            if dmin is not None:
                b[2:4] = struct.pack("<e", dmin)
        return bytes(b)

    equal = bytearray(blk)
    if qtype == T.Q6_K:
        equal[0:128] = bytes([0x55] * 128)   # low nibbles 5
        equal[128:192] = bytes([0x00] * 64)  # high bits 0 -> q = 5 - 32 = -27
        equal[192:208] = bytes([3] * 16)     # all scales 3
    else:
        start = 16 if qtype == T.Q4_K else 48
        equal[start:start + 128] = bytes([0x77] * 128)  # nibbles 7
        if qtype == T.Q5_K:
            equal[16:48] = bytes([0] * 32)
        equal[4:16] = bytes([5] * 4) + bytes([2] * 4) + bytes([5 | (2 << 4)] * 4)  # sc=5, m=2 for all 8 sub-blocks
    nb = width // block_size
    return [("hostile_tiled", bytes(blk) * nb), ("hostile_large_scale", with_scale(blk, 60000.0, 100.0) * nb),
            ("hostile_tiny_scale", with_scale(blk, 6.1e-5, 6.1e-5) * nb), ("all_equal_w", bytes(equal) * nb)]


def quant_rows(w, qtype):
    if qtype == T.F32:
        return np.ascontiguousarray(w, dtype="<f4").tobytes(), w.astype(np.float32)
    q = quantize(w, qtype)
    return q.tobytes(), dequantize(q, qtype).astype(np.float32)


def weight_rows(qtype, width, random_rows, names):
    """(names, raw bytes, dequantized f32 rows, source note) for one dot fixture."""
    if qtype in (T.F32, T.Q4_0, T.Q8_0):
        w = torch.stack(random_rows).numpy().astype(np.float32)
        raw, deq = quant_rows(w, qtype)
        return list(names), raw, deq, "random rows quantized with gguf-py"
    tname, reals = real_rows(qtype, width, 3)
    rows = [("real_%d" % i, r) for i, r in enumerate(reals)] + hostile_kquant_rows(qtype, width)
    raw = b"".join(r for _, r in rows)
    deq = dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(len(rows), -1), qtype).astype(np.float32)
    return [n for n, _ in rows], raw, deq, "real rows of %s + hostile tiled blocks (gguf-py has no K-quant quantizer)" % tname


def gen_dot():
    fx = Fixture("dot")
    fx.meta["source"] = "weights: gguf-py quantize (Q4_0/Q8_0/F32) or real GGUF rows (K-quants) then dequantize; activations: gguf-py Q8_0 quantize then dequantize; reference = float64 dot of the two dequantized rows"
    fx.meta["budget_k"] = 8
    fx.meta["budget_note"] = "budget = k * sqrt(n) * eps * max_abs_term; the Rust kernels sum 32-element block partials then accumulate blocks left to right"
    for qtype in QTYPES:
        for width in ((256, 5120, 6144) if qtype == T.Q5_K else (256, 5120, 17408)):
            rows = [("random", rnd(width, scale=0.05)), ("large_w", rnd(width, scale=100.0)), ("tiny_w", rnd(width, scale=1e-4)),
                    ("all_equal_w", torch.full((width,), 0.03)), ("neg_zero_w", torch.full((width,), -0.0))]
            names, rs = zip(*rows)
            names, w_raw, w_deq, w_note = weight_rows(qtype, width, rs, names)
            n_rows = w_deq.shape[0]
            xs = [rnd(width), rnd(width, scale=30.0), torch.full((width,), 2.0), torch.full((width,), -0.0),
                  torch.cat([torch.tensor([DEN, 1e5, -1e5]), rnd(width - 3)]), rnd(width, scale=0.01), rnd(width, scale=3.0)]
            x = torch.stack(xs[:n_rows]).numpy().astype(np.float32)
            x_raw = quantize(x, T.Q8_0)
            x_deq = dequantize(x_raw, T.Q8_0).astype(np.float32)
            refs, max_terms = [], []
            for i in range(n_rows):
                terms = w_deq[i].astype(np.float64) * x_deq[i].astype(np.float64)
                refs.append(float(terms.sum()))
                max_terms.append(float(np.abs(terms).max()))
            c = "%s_w%d" % (qtype.name, width)
            fx.add(c, kind="dot", ggml_type=qtype.name, ggml_type_id=int(qtype), width=width, rows=list(names), w_source=w_note,
                   n_rows=int(n_rows), w_raw=fx.raw(c, "w_raw", w_raw), w_deq=fx.array(c, "w_deq", w_deq),
                   x_raw=fx.raw(c, "x_raw", x_raw.tobytes()), x_deq=fx.array(c, "x_deq", x_deq), ref_f64=refs, max_abs_term=max_terms)
    fx.write()


def gen_matvec():
    fx = Fixture("matvec")
    fx.meta["source"] = "rows of the dot fixture composed: y[r] = sum_i W_deq[r, i] * x_deq[i]; reference float64"
    fx.meta["budget_k"] = 8
    for qtype in QTYPES:
        for (nrows, width) in ((16, 256), (24, 5120)):
            x = rnd(width).numpy().astype(np.float32)
            if qtype in (T.F32, T.Q4_0, T.Q8_0):
                w = rnd(nrows, width, scale=0.05)
                w[1] = 0.03
                w[2] *= 1e3
                w_raw, w_deq = quant_rows(w.numpy().astype(np.float32), qtype)
                note = "random rows"
            else:
                tname, reals = real_rows(qtype, width, nrows)
                w_raw = b"".join(reals)
                w_deq = dequantize(np.frombuffer(w_raw, dtype=np.uint8).reshape(nrows, -1), qtype).astype(np.float32)
                note = "real rows of " + tname
            x_raw = quantize(x.reshape(1, -1), T.Q8_0)
            x_deq = dequantize(x_raw, T.Q8_0).astype(np.float32).reshape(-1)
            terms = w_deq.astype(np.float64) * x_deq.astype(np.float64)[None, :]
            c = "%s_r%d_w%d" % (qtype.name, nrows, width)
            fx.add(c, kind="matvec", ggml_type=qtype.name, ggml_type_id=int(qtype), n_rows=nrows, width=width, w_source=note,
                   w_raw=fx.raw(c, "w_raw", w_raw), x_raw=fx.raw(c, "x_raw", x_raw.tobytes()), x_deq=fx.array(c, "x_deq", x_deq),
                   ref_f64=[float(v) for v in terms.sum(axis=1)], max_abs_term=[float(v) for v in np.abs(terms).max(axis=1)])
    fx.write()


# --------------------------------------------------------------------------------------------- rope
def rope_cfg(head_dim, sections, partial=0.25):
    return Qwen3_5TextConfig(hidden_size=head_dim * 4, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": partial,
                                              "mrope_section": sections, "mrope_interleaved": True},
                             max_position_embeddings=262144, layer_types=["full_attention"], num_hidden_layers=1)


def gen_rope():
    fx = Fixture("rope")
    fx.meta["source"] = "Qwen3_5TextRotaryEmbedding (84-167) with equal T/H/W position streams + apply_rotary_pos_emb (557-593); see docs/rope.md"
    fx.meta["budget_k"] = 8
    fx.meta["tables_note"] = "cos/sin are computed by torch in float32: angle = float32(pos) * inv_freq (float32 multiply), then cos/sin in float32"
    for head_dim, sections in ((256, [11, 11, 10]), (16, [1, 1, 0])):
        cfg = rope_cfg(head_dim, sections)
        rot = M.Qwen3_5TextRotaryEmbedding(cfg)
        rope_dim = int(head_dim * 0.25)
        positions = torch.tensor([0, 1, 2, 3, 1000, 262143], dtype=torch.long)
        tlen = positions.numel()
        pos3 = positions.view(1, 1, -1).expand(3, 1, -1)
        q = rnd(1, 4, tlen, head_dim)
        k = rnd(1, 2, tlen, head_dim)
        q[0, 0, 0, :8] = torch.tensor([DEN, -DEN, 0.0, -0.0, 1e18, -1e18, 1.0, -1.0])
        k[0, 0, 1, :4] = torch.tensor([1e30, -1e30, DEN, -0.0])
        with torch.no_grad():
            cos, sin = rot(q, pos3)
            qe, ke = M.apply_rotary_pos_emb(q, k, cos, sin)
        c = "rope_hd%d" % head_dim
        fx.add(c, kind="rope", head_dim=head_dim, rope_dim=rope_dim, theta=10000000.0, mrope_section=sections,
               positions=[int(p) for p in positions], inv_freq=fx.array(c, "inv_freq", rot.inv_freq),
               cos=fx.array(c, "cos", cos[0]), sin=fx.array(c, "sin", sin[0]),
               q=fx.array(c, "q", q[0]), k=fx.array(c, "k", k[0]), q_out=fx.array(c, "q_out", qe[0]), k_out=fx.array(c, "k_out", ke[0]),
               max_abs=float(max(qe[torch.isfinite(qe)].abs().max(), ke[torch.isfinite(ke)].abs().max())))
    fx.write()


# --------------------------------------------------------------------------------------------- causal conv1d
def gen_conv():
    fx = Fixture("conv1d")
    fx.meta["source"] = "causal_conv1d_update (200-218): window = cat(state, x_new); out = sum_j w[c, j] * window[j]; state <- window[1:]; then silu. causal_conv1d_fn (220-240) for prefill"
    fx.meta["budget_k"] = 8
    for channels in (16, 10240):
        kernel = 4
        w = rnd(channels, kernel, scale=0.5)
        state = rnd(channels, kernel - 1)
        x_new = rnd(channels, 1)
        state[0] = DEN
        state[1] = 1e18
        x_new[2, 0] = -0.0
        w[3] = 2.0
        x_new[3, 0] = 2.0
        state[3] = 2.0
        st = state.clone().unsqueeze(0)
        with torch.no_grad():
            out = M.causal_conv1d_update(x_new.unsqueeze(0), st, w, None, "silu")
        c = "conv_step_c%d" % channels
        fx.add(c, kind="conv_step", channels=channels, kernel=kernel, w=fx.array(c, "w", w), state=fx.array(c, "state", state),
               x_new=fx.array(c, "x_new", x_new[:, 0]), out=fx.array(c, "out", out[0, :, 0]), new_state=fx.array(c, "new_state", st[0]),
               max_abs=float(out[torch.isfinite(out)].abs().max()))
        tlen = 7
        x = rnd(channels, tlen)
        with torch.no_grad():
            outp = M.causal_conv1d_fn(x.unsqueeze(0), w, None, activation="silu")
        c = "conv_prefill_c%d" % channels
        fx.add(c, kind="conv_prefill", channels=channels, kernel=kernel, t=tlen, w=fx.array(c, "w", w), x=fx.array(c, "x", x),
               out=fx.array(c, "out", outp[0]), max_abs=float(outp.abs().max()), note="zero initial state; T sequential conv_step calls must reproduce this")
    fx.write()


# --------------------------------------------------------------------------------------------- deltanet step
def gen_deltanet():
    fx = Fixture("deltanet")
    fx.meta["source"] = "torch_recurrent_gated_delta_rule (331-386) with use_qk_l2norm_in_kernel=True; gates from Qwen3_5GatedDeltaNet.forward lines 492-494; see docs/deltanet.md"
    fx.meta["budget_k"] = 16
    fx.meta["order"] = "S *= exp(g); kv_mem = k^T S; delta = (v - kv_mem) * beta; S += outer(k, delta); out = q^T S; q,k L2-normed (eps 1e-6) and q scaled by 1/sqrt(dk) BEFORE the step"
    for (heads, dk, dv) in ((6, 8, 8), (3, 128, 128)):
        q = rnd(1, 1, heads, dk)
        k = rnd(1, 1, heads, dk)
        v = rnd(1, 1, heads, dv)
        a = rnd(1, 1, heads, scale=3.0)
        b = rnd(1, 1, heads, scale=3.0)
        A = -torch.exp(rnd(heads, scale=1.5))  # as stored in GGUF: ssm_a = -exp(A_log)
        dt_bias = rnd(heads, scale=0.5)
        state = rnd(1, heads, dk, dv, scale=0.3)
        q[0, 0, 0] = 0.0          # l2norm of a zero vector
        a[0, 0, 1] = 40.0         # softplus above the threshold
        b[0, 0, 1] = -40.0
        state[0, 2] = DEN
        k[0, 0, 2, 0] = -0.0
        if heads > 3:
            a[0, 0, 3] = -40.0
            b[0, 0, 4] = 40.0
            state[0, 5] *= 1e6
        with torch.no_grad():
            beta = b.sigmoid()
            g = A.float() * F.softplus(a.float() + dt_bias)
            out, new_state = M.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        c = "step_h%d_dk%d" % (heads, dk)
        fx.add(c, kind="deltanet_step", heads=heads, dk=dk, dv=dv,
               q=fx.array(c, "q", q[0, 0]), k=fx.array(c, "k", k[0, 0]), v=fx.array(c, "v", v[0, 0]),
               a=fx.array(c, "a", a[0, 0]), b=fx.array(c, "b", b[0, 0]), ssm_a=fx.array(c, "ssm_a", A), dt_bias=fx.array(c, "dt_bias", dt_bias),
               g=fx.array(c, "g", g[0, 0]), beta=fx.array(c, "beta", beta[0, 0]),
               state=fx.array(c, "state", state[0]), out=fx.array(c, "out", out[0, 0]), new_state=fx.array(c, "new_state", new_state[0]),
               max_abs=float(max(out.abs().max(), new_state.abs().max())))
    fx.write()


if __name__ == "__main__":
    from gen_q8k import gen_q8k  # Phase 3: Q8_K activation quantiser (ported reference, gguf-py has none)

    gen_rmsnorm()
    gen_act()
    gen_softmax()
    gen_q8()
    gen_q8k()
    gen_dot()
    gen_matvec()
    gen_rope()
    gen_conv()
    gen_deltanet()
