"""Shared helpers for the Phase 2 fixture generators.

- `Fixture`: collects named float32 arrays (saved as .npy, C order, '<f4') and JSON metadata into one
  manifest per generator: tests/fixtures/kernels/<group>/manifest.json (or layers/).
- `reorder_v_heads`: byte-for-byte copy of llama.cpp `conversion/qwen.py::_LinearAttentionVReorderBase._reorder_v_heads`
  (commit 67a17c1) so layer fixtures are emitted in GGUF tensor layout. `check_reorder_matches_converter`
  asserts equality against the real converter when models/llama.cpp is present.
- `gguf_layout_deltanet`: applies every converter transform for one DeltaNet block (docs/deltanet.md table).
- `gguf_layout_norm`: the +1 for zero-centered norms.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
F32_EPS = float(np.finfo(np.float32).eps)  # 2^-23


def to_np(t):
    return np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype="<f4")


class Fixture:
    def __init__(self, group, subdir="kernels"):
        self.dir = os.path.join(FIXTURES, subdir, group)
        os.makedirs(self.dir, exist_ok=True)
        self.cases = []
        self.meta = {"group": group, "torch": torch.__version__, "f32_eps": F32_EPS}

    def array(self, case, name, arr):
        arr = arr.detach().cpu() if isinstance(arr, torch.Tensor) else arr
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32), dtype="<f4")
        fname = "%s__%s.npy" % (case, name)
        np.save(os.path.join(self.dir, fname), arr)
        return {"file": fname, "shape": list(arr.shape)}

    def raw(self, case, name, b):
        fname = "%s__%s.bin" % (case, name)
        open(os.path.join(self.dir, fname), "wb").write(bytes(b))
        return {"file": fname, "bytes": len(b)}

    def add(self, case, **fields):
        entry = {"case": case}
        entry.update(fields)
        self.cases.append(entry)
        return entry

    def write(self):
        self.meta["cases"] = self.cases
        text = json.dumps(self.meta, indent=1, allow_nan=False)
        open(os.path.join(self.dir, "manifest.json"), "w").write(text)
        print("wrote %s (%d cases)" % (os.path.join(self.dir, "manifest.json"), len(self.cases)))


def finite_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(a))) if a.size else 0.0}


def reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Reorder V heads from grouped (by K head) to tiled order along the given dimension.
    Copy of llama.cpp conversion/qwen.py lines 458-469 (commit 67a17c1)."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def check_reorder_matches_converter():
    """If the llama.cpp checkout is present, assert our copy equals the converter's staticmethod."""
    src = os.path.join(ROOT, "models", "llama.cpp", "conversion", "qwen.py")
    if not os.path.exists(src):
        return "converter not present; reorder_v_heads unverified against source"
    text = open(src, encoding="utf-8").read()
    start = text.index("def _reorder_v_heads(")
    body = text[start:text.index("\n\n", start)]
    ns = {"Tensor": torch.Tensor}
    exec("import torch\n" + "\n".join(l[4:] if l.startswith("    ") else l for l in body.splitlines()), ns)
    t = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2 * 3 * 4, 5)
    a = ns["_reorder_v_heads"](t, 0, 2, 3, 4)
    b = reorder_v_heads(t, 0, 2, 3, 4)
    assert torch.equal(a, b), "reorder_v_heads differs from the converter"
    return "reorder_v_heads verified against models/llama.cpp/conversion/qwen.py"


def gguf_layout_deltanet(sd, prefix, num_k, num_v, head_k, head_v):
    """Convert one HF Qwen3_5GatedDeltaNet state dict (keys without the module prefix) to GGUF-layout f32
    tensors named like the GGUF (docs/deltanet.md). Returns dict name -> torch tensor (numpy order).
    Mirrors conversion/qwen.py lines 385-397 and 549-612."""
    r = num_v // num_k
    W = lambda k: sd[prefix + k].detach().float()
    qkv = W("in_proj_qkv.weight")  # (2*kd + vd, hidden)
    kd = num_k * head_k
    q, k, v = qkv[:kd], qkv[kd:2 * kd], qkv[2 * kd:]
    v = reorder_v_heads(v, 0, num_k, r, head_v)
    out = {
        "attn_qkv.weight": torch.cat([q, k, v], dim=0),
        "attn_gate.weight": reorder_v_heads(W("in_proj_z.weight"), 0, num_k, r, head_v),
        "ssm_alpha.weight": reorder_v_heads(W("in_proj_a.weight"), 0, num_k, r, 1),
        "ssm_beta.weight": reorder_v_heads(W("in_proj_b.weight"), 0, num_k, r, 1),
        "ssm_a": reorder_v_heads((-torch.exp(W("A_log"))).unsqueeze(-1), 0, num_k, r, 1).squeeze(-1),
        "ssm_dt.bias": reorder_v_heads(W("dt_bias").unsqueeze(-1), 0, num_k, r, 1).squeeze(-1),
        "ssm_norm.weight": W("norm.weight"),  # not zero-centered: no +1
    }
    conv = W("conv1d.weight").squeeze()  # (conv_dim, kernel)
    qk = conv[: 2 * kd]
    vpart = reorder_v_heads(conv[2 * kd:], 0, num_k, r, head_v)
    out["ssm_conv1d.weight"] = torch.cat([qk, vpart], dim=0)
    out["ssm_out.weight"] = reorder_v_heads(W("out_proj.weight"), 1, num_k, r, head_v)
    return out


def gguf_layout_norm(w):
    """Zero-centered Qwen3_5RMSNorm weight as stored in the GGUF (converter adds 1)."""
    return w.detach().float() + 1.0


def budget(k, width, max_abs):
    """Rule 5: k * sqrt(width) * f32_eps * max|activation|."""
    return float(k) * float(np.sqrt(width)) * F32_EPS * float(max_abs)
