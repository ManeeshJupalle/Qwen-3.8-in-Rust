# RoPE in Qwen3.5 (captured from the HF reference before writing the kernel)

Source: `.venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.16.1).
Line numbers below refer to that file. Config values are from `tests/fixtures/config.json` and the GGUF
metadata (`docs/config-mapping.md`).

## Which layers, which dims

- Only the 16 full-attention layers (`blk.3, 7, ..., 63`) apply RoPE. `Qwen3_5GatedDeltaNet` (lines 387-548)
  receives no position embeddings; DeltaNet layers have no position parameters at all.
- **Partial rotary.** `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters` (lines 106-125):
  `dim = int(head_dim * partial_rotary_factor) = int(256 * 0.25) = 64`, matching the GGUF key
  `qwen35.rope.dimension_count = 64`. `inv_freq[i] = 1 / rope_theta ** (2i / 64)` for `i` in `0..32`,
  `rope_theta = 10_000_000` (`qwen35.rope.freq_base`). `attention_scaling = 1.0` (rope_type `default`, no YaRN).
- `apply_rotary_pos_emb` (lines 557-593): `rotary_dim = cos.shape[-1] = 64`. `q_rot = q[..., :64]`,
  `q_pass = q[..., 64:]` (192 dims untouched). `q_embed = q_rot * cos + rotate_half(q_rot) * sin`, then
  `cat([q_embed, q_pass])`. Same for k. **The first 64 of the 256 head dims rotate; dims 64..256 pass through.**
- `rotate_half` (lines 549-554): splits the 64 rotary dims into two halves of 32 and returns `cat(-x2, x1)`.
  So dim `i` pairs with dim `i + 32` (NeoX / non-interleaved layout), for `i` in `0..32`:
  `out[i] = x[i] * cos_i - x[i+32] * sin_i`, `out[i+32] = x[i+32] * cos_i + x[i] * sin_i`, with
  `cos_i = cos(pos * inv_freq[i])` because `emb = cat(freqs, freqs)` (line 148) duplicates the 32 frequencies.

## Where RoPE sits in the attention block (lines 664-687)

1. `q_proj(x)` -> view `(.., heads, 2 * head_dim)` -> `chunk(2, dim=-1)` = `(query, gate)`: **per head, the
   first 256 outputs are q and the next 256 are the output gate** (head-major interleave). The gate is applied
   after attention: `attn_output * sigmoid(gate)` (line 704).
2. `q_norm` / `k_norm` (`Qwen3_5RMSNorm` over head_dim, zero-centered: `x * rsqrt(mean(x^2) + eps) * (1 + w)`)
   are applied **before** RoPE (lines 676-677).
3. RoPE on q and k (line 681), then KV-cache update, then attention with `scaling = head_dim ** -0.5`
   (line 640) and a float32 softmax (line 618).

## mRoPE for text-only input

`Qwen3_5TextModel.forward` (lines 1173-1185): when `position_ids` is not given it builds
`arange(T) + past_seen_tokens`, views it `(1, 1, T)` and expands to `(4, B, T)`; stream 0 is the text
position (used for the causal mask), streams 1..3 (`position_ids[1:]`, shape `(3, B, T)`) go to the rotary
module. **All three rotary streams carry the same integer positions for text.**

`Qwen3_5TextRotaryEmbedding.forward` (lines 129-152): `freqs[s, b, t, i] = pos[s, b, t] * inv_freq[i]`
(float32, shape `(3, B, T, 32)`), then `apply_interleaved_mrope(freqs, mrope_section)` (lines 154-166):
starting from the T stream, it copies the H stream into indices `1, 4, 7, ..., 31` (`slice(1, 3*11, 3)`)
and the W stream into `2, 5, ..., 29` (`slice(2, 3*10, 3)`), i.e. the 32 frequency slots are interleaved
T,H,W,T,H,W,... with the last two slots T-only. `mrope_section = [11, 11, 10]` sums to 32 = rope_dim / 2;
the GGUF's `qwen35.rope.dimension_sections = [11, 11, 10, 0]` has a fourth (unused) entry.
**With equal positions the copies are no-ops, so text-only mRoPE is exactly plain RoPE with theta 1e7 on
64 dims.** `mrope_interleaved = true` (arch constant) therefore has no effect on any Phase 2 kernel; it is
recorded so a future image/video path knows the slot layout.

## Fixture

`tools/fixtures/gen_kernels.py::rope` builds `Qwen3_5TextRotaryEmbedding` from a `Qwen3_5TextConfig` with
the real values (head_dim 256, partial 0.25, theta 1e7, sections [11, 11, 10]) and a tiny variant
(head_dim 16, rope_dim 4, sections [1, 1, 0]), feeds `position_ids` of shape `(3, 1, T)` with equal streams,
and applies `apply_rotary_pos_emb` to random q/k tiles (`(1, heads, T, head_dim)`). Positions include 0,
1, and 262143 (max context). The Rust kernel `kernels::rope::apply_rope` takes `(heads, head_dim)` tiles,
`rope_dim`, `theta`, and an integer position, and must match within the derived budget
(`k * sqrt(2) * eps * max|x|` per output element: each output is a 2-term sum).

A second fixture with different T/H/W positions is NOT generated: the engine is text-only in v1 and the
kernel does not implement the interleave; `mrope_interleaved` is an assertion, not a code path.
