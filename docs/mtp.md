# The MTP draft head (Phase 5.1): what `blk.64` computes, captured before writing `layers/mtp.rs`

Written from three implementations that all serve this checkpoint's MTP head, read side by side, and from the
Phase 0 tensor capture. Nothing here is inferred from a config field.

Sources (all fetched or checked out on 2026-09-04):

- vLLM `vllm/model_executor/models/qwen3_next_mtp.py` (`Qwen3NextMultiTokenPredictor.forward`, lines 99-137 of
  the current main) and `vllm/model_executor/models/qwen3_next.py` (`Qwen3NextModel.forward`, return at line 107;
  `Qwen3NextRMSNorm = GemmaRMSNorm`, `layernorm.py` line 157: `weight = self.weight.float() + 1.0`);
  the proposer `vllm/v1/spec_decode/llm_base_proposer.py` (`propose`, lines 510-830; the input shift at lines
  848-859; `model_returns_tuple` lines 1016-1028) and `vllm/v1/worker/gpu_model_runner.py` (the hidden state handed
  to the drafter, line 5314). vLLM has no separate Qwen3.5 MTP file: `Qwen3_5` text models reuse the
  `Qwen3Next` MTP module (the `models/` directory holds `qwen3_next_mtp.py` and no `qwen3_5_mtp.py`).
- SGLang `python/sglang/srt/models/qwen3_next_mtp.py` (`Qwen3NextForCausalLMMTP.forward`, lines 105-136;
  `GemmaRMSNorm` for the three norms, one `Qwen3NextModel` layer with `full_attention_interval = 1`).
- llama.cpp `src/models/qwen35.cpp` at commit 67a17c1 (`models/llama.cpp`): the main graph's `t_h_nextn`
  (lines 205-209) and the `graph_mtp` draft graph (lines 485-640); the driver `common/speculative.cpp`
  (`common_speculative_impl_draft_mtp`, lines 1324-1745).
- The converter `conversion/qwen.py` (`_QwenMtpMixin.filter_tensors` lines 320-347, `modify_tensors` line 394).
- Phase 0: `docs/data/gguf_layout.txt` layer 64, `docs/payload-vs-doc.md` findings 3, 20, 23; the tiny model's
  `blk.9` (`docs/data/tiny_layout.txt`).

HF transformers 5.16.1 is **not** a reference: `modeling_qwen3_5.py` lists `^mtp.*` under
`_keys_to_ignore_on_load_unexpected` (lines 807, 1584) and implements nothing for it.

## The tensors (bartowski `Qwen3.8-27B-Q4_K_M.gguf`, `blk.64`, 238,983,168 bytes contiguous)

| GGUF tensor | HF tensor (safetensors) | shape (numpy) | type | role |
|---|---|---|---|---|
| `blk.64.nextn.eh_proj.weight` | `mtp.fc.weight` | [5120, 10240] | Q4_0 | the projection of `[enorm(embed) \| hnorm(hidden)]` to hidden |
| `blk.64.nextn.enorm.weight` | `mtp.pre_fc_norm_embedding.weight` | [5120] | F32 | RMSNorm of the token embedding, **+1 folded in** |
| `blk.64.nextn.hnorm.weight` | `mtp.pre_fc_norm_hidden.weight` | [5120] | F32 | RMSNorm of the target hidden state, +1 folded in |
| `blk.64.attn_norm.weight`, `post_attention_norm.weight` | `mtp.layers.0.input_layernorm`, `post_attention_layernorm` | [5120] | F32 | the block's two norms, +1 folded in |
| `blk.64.attn_q.weight` | `mtp.layers.0.self_attn.q_proj` | [12288, 5120] | Q4_0 | `[q \| gate]` per head, as in the 16 main GQA layers |
| `blk.64.attn_k.weight`, `attn_v.weight` | `k_proj`, `v_proj` | [1024, 5120] | Q4_0 | 4 KV heads x 256 |
| `blk.64.attn_q_norm.weight`, `attn_k_norm.weight` | `q_norm`, `k_norm` | [256] | F32 | per-head RMSNorm, +1 folded in |
| `blk.64.attn_output.weight` | `o_proj` | [5120, 6144] | Q4_0 | |
| `blk.64.ffn_gate/up.weight`, `ffn_down.weight` | `mlp.gate_proj/up_proj`, `down_proj` | [17408, 5120], [5120, 17408] | Q4_0 | SwiGLU MLP |
| `blk.64.nextn.shared_head_norm.weight` | `mtp.norm.weight` | [5120] | F32 | the head norm, +1 folded in |
| (none) | (none) | | | no `nextn.embed_tokens`, no `nextn.shared_head_head`: the draft uses `token_embd.weight` and `output.weight` |

**Layer type: GQA (full attention), not DeltaNet.** The block has `attn_q/k/v/output` and `attn_q_norm/k_norm`
and no `ssm_*` or `attn_qkv/attn_gate` tensors (Phase 0's guess, now confirmed by the shapes and by every
reference: vLLM builds it as `Qwen3NextDecoderLayer(layer_type="full_attention")`, SGLang with
`full_attention_interval = 1`, llama.cpp's `graph_mtp` runs `build_attn` with a KV cache).

**The +1 on every nextn norm.** The converter renames `mtp.pre_fc_norm_embedding.weight` to
`model.layers.64.enorm.weight` (and `hnorm`, `shared_head.norm`, `layers.0.*` likewise) *before*
`modify_tensors`, whose rule `name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight") ->
data_torch + 1` therefore applies to all of them. Verified on the tiny model, whose GGUF went through the same
converter: `blk.9.nextn.enorm/hnorm/shared_head_norm/attn_norm/attn_q_norm` equal the safetensors values + 1
exactly (max diff 0), while `eh_proj` and `attn_q` are the safetensors values unchanged. So the engine uses the
stored weights with its ordinary `rmsnorm` (`y = x * rsqrt(mean(x^2) + eps) * w`), like every other norm;
the references' `(1 + weight)` is already inside `w`. eps is the model's `rms_norm_eps` (1e-6) everywhere.

## The draft step (one MTP row)

Inputs for row `j`: a token id `x` and a target hidden state `h` (5120 f32). Output: the draft logits and the
MTP's own hidden `h'` for the next chained row.

1. `e = token_embd[x]` (the shared embedding row, dequantised to f32; `mtp_use_dedicated_embeddings = false`).
2. `en = rmsnorm(e, enorm)`; `hn = rmsnorm(h, hnorm)`.
3. `c = [en | hn]` (10240; **embedding first, hidden second**: vLLM `torch.cat([inputs_embeds, hidden_states], dim=-1)`,
   SGLang `torch.cat((input_embeds, hidden_states), dim=-1)`, llama.cpp `ggml_concat(e_norm, h_norm, 0)`).
4. `u = eh_proj . c` (5120; Q4_0 row x Q8_0 activation, the kernels of Phase 3).
5. One decoder block exactly as the 16 main GQA layers (`layers::DecoderLayer` with `Mixer::Attention`, so the
   Phase 2/3 code and fixtures cover it): `a = attn_norm(u)`; `[q | gate] = attn_q . a` per head; `q_norm`, `k_norm`
   per head; partial RoPE on the first 64 dims at the row's position; scores over the MTP's own KV cache scaled by
   `256^-0.5`; softmax; V sum; `* sigmoid(gate)`; `attn_output`; residual `u + attn`; `post_attention_norm`;
   SwiGLU MLP; residual. (llama.cpp lines 551-619 spell out the same ops in ggml; vLLM/SGLang run their
   `Qwen3NextDecoderLayer`.)
6. `h' = rmsnorm(block_out, shared_head_norm)`.
7. `logits = output.weight . h'` (the shared lm_head, Q6_K x Q8_K); the draft token is `argmax` (greedy) or a
   sample from the processed distribution (5.3).

**What the next chained row consumes.** Row `j + 1` takes `x = the draft just produced` and `h = h'` from step 6,
i.e. the MTP's *post-`shared_head_norm`* output, not the pre-norm block output: vLLM's proposer sets
`hidden_states = last_hidden_states = ret_hidden_states` for method `mtp` unless the architecture is in the
tuple-returning set (`DeepSeekMTPModel`, `DeepseekV32MTPModel`, `Glm5NextMTPModel`, `KimiK3MTPModel`;
`Qwen3NextMTP` is not), and `Qwen3NextMultiTokenPredictor.forward` returns `self.norm(hidden_states, residual)`;
llama.cpp's `graph_mtp` sets `res->t_h_nextn` after `build_norm(cur, shared_head_norm)` and the driver feeds
`llama_get_embeddings_nextn_ith(ctx_dft, i_last)` into the next draft row (speculative.cpp lines 1666, 1717).

**Which target hidden `h` is.** The *post-final-norm* hidden of the main model, `output_norm(h_64)`, the vector
the lm_head reads: vLLM's `Qwen3NextModel.forward` returns `hidden_states` after `self.norm` (line 107) and the
runner hands exactly that to the drafter (`target_hidden_states = hidden_states[:num_scheduled_tokens]`, line
5314; `get_mtp_target_hidden_states` is an override only DeepSeek V4 uses); SGLang's model returns
`self.norm(hidden_states, residual)`; llama.cpp's main graph assigns `res->t_h_nextn = cur` *after*
`build_norm(cur, model.output_norm)` (qwen35.cpp lines 205-209; the field's comment in `llama-graph.h` says
"before final output norm" and the code says after). So `hnorm` is applied to an already-normed vector; that is
what all three serve and what the engine does. In the engine this is `State::lm_normed` after `logits_with`, so
no extra work is needed to obtain it.

## Positions, the cache, and the prompt

The MTP block is GQA, so it carries a **KV cache of its own** across rows (4 KV heads x 256 x 2 x f32 = 8,192
bytes per row). Row `j` of the MTP holds the pair `(x_{j+1}, h_j)`: the token that *follows* position `j` of the
main sequence, paired with the main model's hidden *at* position `j`; it predicts `x_{j+2}`. This is vLLM's input
shift ("`[a1, b1, b2, ...] -> [b1, b2, ...]`, replace the last token with the next token", lines 855-859: input
`i` is `target_token_ids[i + 1]`, the last is the sampled token, positions unchanged) and the DeepSeek-V3 MTP
formulation the head was trained with (`h_i` with `emb(t_{i+1})` predicts `t_{i+2}`).

- **RoPE position of row `j` is `j`** (vLLM: `positions = target_positions`). llama.cpp numbers the same row
  `j + 1` (it pairs token `x_p` at its own position `p` with `h_{p-1}`, speculative.cpp lines 1521-1527, 1661) and
  additionally feeds a row `(x_0, 0)` at position 0 (`pending_h` starts as zeros, line 1434). Since text RoPE is
  relative, a uniform shift of every position changes nothing; the zero row is an extra key the trained
  formulation never had. The engine follows vLLM: no zero row, row index = position = KV cache length before
  the row is appended.
- **The prompt fills the cache** (5.1 asked to confirm rather than assume): both drivers run the MTP layer over
  every prompt position. vLLM's first `propose` after prefill takes `target_token_ids = prompt[1..T] +
  [sampled]`, `target_positions = 0..T-1`, `target_hidden_states = h_0..h_{T-1}`: `T` rows batched through the
  block, whose last row yields draft 1. llama.cpp's `process()` hook decodes the MTP over every target batch,
  prefill included, with the hidden rows shifted right by one (lines 1524-1527). The engine does the same with
  the batched attention path of Phase 3.3 (`GqaAttention::forward_prefill`), one weight read for the whole
  prompt.
- **After a verification round** (accepted drafts `d_1..d_m` for positions `P+1..P+m`, bonus `x_{P+m+1}`, target
  hiddens `h_P..h_{P+m}` from the verify batch): the MTP cache is **truncated to `P` rows** (rows `P..` were
  written during drafting with the MTP's own chained hiddens) and `m + 1` rows `(x_{P+1}, h_P), ...,
  (x_{P+m+1}, h_{P+m})` are fed as one batch; the last row's logits are the next round's draft 1, and drafts
  2..k chain from `h'`. This is vLLM's next `propose` (`target_token_ids` = the accepted tokens plus the bonus,
  hidden states from the verify forward at those positions) and llama.cpp's `process()` on the verify batch
  followed by `llama_memory_seq_rm` of the rejected tail. Position bookkeeping in `docs/spec.md`.

## Accuracy: drafts need to be likely, not exact

bartowski stores the whole MTP block as Q4_0 (finding 23: the noisiest of the three copies, 9 % relative RMS
against Unsloth's Q6_K/Q8_0, `docs/data/mtp_blk64_diff.txt`), and the engine feeds it Q8_0 activations. None of
that touches the output: with exact greedy verification (`docs/spec.md`) every emitted token is the main model's
argmax whatever the draft was, and with sampling the acceptance rule is distribution-preserving for any draft
distribution. Draft quality only moves the acceptance rate, which 5.4 measures rather than assumes. The engine
therefore does not dequantise the block to anything finer.

## Fixture (`tools/ref_mtp.py`, `tests/fixtures/mtp/`)

The tiny checkpoint carries a random MTP block (finding 29: `mtp.*` in `models/tiny/model.safetensors`,
`blk.9.*` in the GGUF, same converter as the real file). The reference implements the step above in PyTorch on
the HF side (safetensors weights, `Qwen3_5RMSNorm` = `(1 + w) * normed`, `Qwen3_5DecoderLayer` of type
`full_attention` for the block, `Qwen3_5ForCausalLM` for the target hiddens and embeddings), and for each of the
3 fixture prompts: feeds the `T` prompt rows `(x_{j+1}, h_j)` (with `x_T` = the target's first greedy token,
`h_j` = the target's post-final-norm hidden at `j`, i.e. the tiny fixture's `final_norm` array), takes draft 1 from
the last row, then chains 19 more drafts from `h'`. Filed per prompt: the draft-1 logits (full vocab), the
20 draft ids, the top-5 ids and logits and the top-1/top-2 margin at every step, and `max|logits|` for the
budget. The Rust test (`tests/mtp_oracle.rs`) loads the tiny GGUF, runs `layers::mtp` the same way and requires:
draft-1 logits within the tiny oracle's budget (`k * (n_layer + 2) * sqrt(hidden) * eps * max|logits|`, k = 16,
one block on top of the 9-layer target), the top-1 logit at every step within the same budget, and 20/20 draft
ids on all 3 prompts (60/60 in the gate).

## Cost per draft (for the cost model of 5.4)

One MTP row: `eh_proj` 29.5 MB + the block 209 MB (Q4_0, read from RAM: the block is always resident) + the
head 1,043 MB (Q6_K) = 1.28 GB through the CPU, so `c_mtp` is about 1.28 GB / membw plus the block's
non-matvec work; at 30 GB/s that is roughly 45 ms, dominated by the lm_head. The MTP's batched re-feed of the
`m + 1` accepted rows costs one more block pass (its head is read once, for the last row only).

**Measured (Phase 5.4, `docs/data/spec_cost_model.txt`):** a chained draft step is 0.064 s cool and 0.085 to 0.11 s
throttled (mean over the ladder 0.083 s), the re-feed of `m + 1` rows plus one head pass 0.11 to 0.19 s. The Q4_0
block runs its matvecs at 26 to 40 % of membw (finding 51), which is why the step is nearly twice the head-only
estimate: the block is 16 % of the bytes and about half the time.
