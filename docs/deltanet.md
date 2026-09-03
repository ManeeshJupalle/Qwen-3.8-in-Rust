# Gated DeltaNet in Qwen3.5: exact op order (captured before writing the kernel)

Source: `.venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.16.1),
class `Qwen3_5GatedDeltaNet` lines 387-548, `torch_recurrent_gated_delta_rule` lines 331-386,
`torch_chunk_gated_delta_rule` lines 249-329, `causal_conv1d_update` lines 200-218, `causal_conv1d_fn`
lines 220-240, `l2norm` lines 242-248, `Qwen3_5RMSNormGated` lines 168-186. GGUF layout facts come from
llama.cpp `conversion/qwen.py` (commit 67a17c1, 2026-09-03): `Qwen3NextModel.modify_tensors` lines 385-397
and `_LinearAttentionVReorderBase` lines 446-612.

Real sizes (`docs/config-mapping.md`): hidden 5120; K heads 16 x 128; V heads 48 x 128 (3 V heads per K head);
conv kernel 4; `conv_dim = 2 * 2048 + 6144 = 10240`; eps 1e-6.

## Forward, one layer, sequence of T tokens (module lines 428-547)

Inputs: `x` (T, 5120) = `input_layernorm(h)` (the decoder layer applies the zero-centered RMSNorm first).

1. **Projections** (lines 441-449; all `nn.Linear` without bias):
   `mixed_qkv = in_proj_qkv(x)` -> (T, 10240) = `[q (2048) | k (2048) | v (6144)]`;
   `z = in_proj_z(x)` -> (T, 6144) viewed (T, 48, 128);
   `b = in_proj_b(x)` -> (T, 48); `a = in_proj_a(x)` -> (T, 48).
2. **Causal depthwise conv + SiLU on `mixed_qkv`** (lines 451-476), over time, per channel, kernel 4,
   no bias. `causal_conv1d_fn`: left-pad 3 zeros, `out[t] = sum_{j=0..3} w[c, j] * x[t - 3 + j]`
   (PyTorch conv1d is cross-correlation: the newest sample multiplies `w[3]`), then `silu`.
   Incremental form `causal_conv1d_update`: window = `cat(state[c, 0..3], x_new[c])`, `out = sum_j w[c, j] * window[j]`,
   `state <- window[1..4]`, then `silu`. Activation name is `config.hidden_act = "silu"` (line 400).
3. **Split and reshape** (lines 478-490): `q` (T, 16, 128), `k` (T, 16, 128), `v` (T, 48, 128).
4. **Gates** (lines 492-494), float32:
   `beta = sigmoid(b)` (T, 48); `g = -exp(A_log) * softplus(a + dt_bias)` (T, 48).
   In the GGUF `ssm_a` already stores `-exp(A_log)` (converter line 386-387), so the engine computes
   `g = ssm_a[h] * softplus(a[h] + ssm_dt_bias[h])`. `softplus(x) = log1p(exp(x))` (PyTorch threshold 20:
   returns `x` when `x > 20`).
5. **Head broadcast** (lines 495-497): `q, k = repeat_interleave(3, dim=heads)`: V head `j` uses K head `j // 3`
   in HF order. (GGUF order differs, see below.)
6. **Recurrence** (`torch_recurrent_gated_delta_rule`, lines 331-386, used for single-token decode; the chunk
   form is mathematically the same and is what prefill uses): all in float32,
   - `q = l2norm(q)`, `k = l2norm(k)` with `l2norm(x) = x * rsqrt(sum(x^2) + 1e-6)` (per head, over 128), lines 344-346.
   - `q = q * (1 / sqrt(128))` (line 353; `scale = 1 / (k_head_dim ** 0.5)`).
   - state `S` per head: (128 k, 128 v), carried; zero if no cache.
   - per token (lines 365-376), per V head `h` with its K head:
     1. `S = S * exp(g[h])`                       (decay)
     2. `kv_mem[v] = sum_k S[k, v] * k_t[k]`     (read with the key)
     3. `delta = (v_t - kv_mem) * beta[h]`        (delta write, rank one)
     4. `S = S + outer(k_t, delta)`               i.e. `S[k, v] += k_t[k] * delta[v]`
     5. `out[v] = sum_k S[k, v] * q_t[k]`         (read with the query)
   Op order is decay -> read -> write -> read, exactly as ARCHITECTURE.md assumed.
7. **Gated RMSNorm per head** (lines 540-543, `Qwen3_5RMSNormGated` lines 175-185): on `out` (T*48, 128) with
   gate `z` (T*48, 128): `y = weight * (out * rsqrt(mean(out^2) + eps))`, then `y = y * silu(z)`.
   `weight` (128) is **not** zero-centered (init ones; converter line 393 excludes `linear_attn.norm.weight`
   from the +1), so the engine uses `ssm_norm` as stored. Note the order: norm, multiply by weight, then gate.
8. `output = out_proj(y.reshape(T, 6144))` -> (T, 5120) (line 546).

The decoder layer then adds the residual, applies `post_attention_layernorm`, MLP, residual (lines 767-797).

## GGUF layout vs HF layout (converter `conversion/qwen.py`)

| GGUF tensor | HF tensor | transform on conversion |
|---|---|---|
| `blk.N.attn_qkv.weight` [5120, 10240] | `in_proj_qkv` | q, k rows unchanged; **v rows re-tiled** (see below) |
| `blk.N.attn_gate.weight` [5120, 6144] | `in_proj_z` | rows re-tiled |
| `blk.N.ssm_alpha.weight` [5120, 48] | `in_proj_a` | rows re-tiled |
| `blk.N.ssm_beta.weight` [5120, 48] | `in_proj_b` | rows re-tiled |
| `blk.N.ssm_a` [48] | `A_log` | `-exp(A_log)`, re-tiled |
| `blk.N.ssm_dt.bias` [48] | `dt_bias` | re-tiled |
| `blk.N.ssm_conv1d.weight` [4, 10240] (numpy [10240, 4]) | `conv1d.weight` [10240, 1, 4] | squeezed; v channels re-tiled |
| `blk.N.ssm_norm.weight` [128] | `norm.weight` | unchanged (no +1) |
| `blk.N.ssm_out.weight` [6144, 5120] (numpy [5120, 6144]) | `out_proj` | **input columns** re-tiled |
| `blk.N.attn_norm.weight`, `post_attention_norm.weight` | `input_layernorm`, `post_attention_layernorm` | `+ 1` (zero-centered norms) |

**V-head re-tiling** (`_reorder_v_heads`, lines 458-469): HF stores V heads grouped by K head,
`j = k * 3 + v` (K head `k` in 0..16, `v` in 0..3). The GGUF stores them tiled: position `p = v * 16 + k`.
Consequently, in GGUF order, **V head `p` uses K head `p % 16`**, and `g`, `beta`, `z`, `A`, `dt_bias`
indexed by `p` line up with the re-tiled rows. The engine works in GGUF order throughout; it never
reproduces the HF order. `tools/fixtures/common.py::reorder_v_heads` copies the converter's function so
layer fixtures are emitted in GGUF layout, and `tools/make_tiny_checkpoint.py` runs the real converter.

## State per layer (f32, arch constant `recurrent_state_dtype = F32`)

- recurrent state: 48 heads x 128 x 128 = 786,432 f32 (3 MiB) per DeltaNet layer, 48 layers -> 144 MiB;
- conv window: 10240 channels x 3 previous samples = 30,720 f32 per layer.
Both are carried across tokens; prefill of T tokens equals T sequential steps (the chunk kernel is an
algebraic rearrangement of the same recurrence).

## What the kernels implement

- `kernels::conv::causal_conv1d_step(state[c][3], w[c][4], x_new[c]) -> (out[c], new state)`, SiLU applied by the caller flag.
- `kernels::deltanet::step(q[128], k[128], v[128], g, beta, S[128*128]) -> out[128]` (one head; q/k already
  L2-normed and q scaled by the caller, per the order above; the kernel documents its summation order:
  `kv_mem[v]` and `out[v]` are plain left-to-right sums over k).
- `kernels::rmsnorm::rmsnorm_gated` for step 7.
Parallelism is over V heads (48 independent states).

## Verified (Phase 2a, `docs/data/phase2a_tests.txt`)

`crates/core/tests/kernels.rs::deltanet_step_matches_hf` and `tests/layers.rs::gated_deltanet_prefill_and_step`
reproduce `torch_recurrent_gated_delta_rule` and the full `Qwen3_5GatedDeltaNet` module (prefill through the
chunk kernel, then a cached step) within k = 16 budgets; measured max diffs are ~4e-7 on outputs of order 1.
HF's chunked prefill and its sequential recurrence differ from each other by 4.6e-7 (fixture field
`chunk_vs_seq_max_diff`). HF's cache keeps 4 conv samples per channel; only the last 3 are used.

## Reference side (Phase 2b): reading the GGUF layout back into the HF module

`tools/fixtures/common.py::hf_layout_deltanet` inverts the table above so `tools/ref_forward.py --weights gguf`
can load dequantised GGUF tensors into `Qwen3_5GatedDeltaNet`: V heads are un-tiled with `reorder_v_heads`
called with the two head counts exchanged (position `p = v * 16 + k` back to `j = k * 3 + v`), on the V rows
of `attn_qkv`, on `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, the V channels of
`ssm_conv1d` and the input columns of `ssm_out`; `conv1d.weight` gets its middle axis back; every zero-centered
norm loses the +1; `ssm_norm` is copied. `A_log` is recovered as the float32 value whose `-exp` round-trips to
the stored `ssm_a`, searching up to three ulps around `log(-ssm_a)`: on the real file only 17 of the 48 DeltaNet
layers round-trip exactly on all 48 heads, the rest keep a 1-ulp residual on some heads (finding 32). The
inverse is verified two ways: `check_gguf_hf_roundtrip` (HF -> GGUF -> HF is the identity on random shapes) and
`tools/ref_gguf_selftest.py` (the tiny F32 GGUF read through the inverse reproduces the tiny HF reference
bit for bit on every layer and the logits).
