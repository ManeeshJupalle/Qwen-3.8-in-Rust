# Payload vs. documentation: where the downloaded files disagree with the model card, the vLLM recipe, the Unsloth guide, or ARCHITECTURE.md

Every entry cites the file it was read from. Evidence files live in `tests/fixtures/` and `docs/data/`.
"Doc" means: the HF model card (`models/Qwen3.8-27B/README.md`), the vLLM recipe
(`docs/data/vllm_recipe_extract.txt`), the Unsloth guide (`docs/data/unsloth_guide_extract.txt`),
or the "Model facts" section of `ARCHITECTURE.md`.

## 1. The engine's kernel plan assumes one weight format; the file has nine

ARCHITECTURE.md: "Weight format: Q4_K (super-blocks of 256, 6-bit scales/mins) as shipped in Q4_K_M GGUF."

Payload (`docs/data/gguf_layout.txt`, "bytes per ggml type"): the file `Qwen3.8-27B-UD-Q4_K_M.gguf` contains

| ggml type | tensors | bytes | share |
|---|---|---|---|
| Q5_K | 131 | 4,833,607,680 | 29.4% |
| IQ4_XS | 117 | 4,760,043,520 | 28.9% |
| Q4_K | 104 | 4,203,970,560 | 25.6% |
| Q6_K | 30 | 1,812,787,200 | 11.0% |
| IQ4_NL | 7 | 330,301,440 | 2.0% |
| Q3_K | 7 | 268,083,200 | 1.6% |
| IQ3_S | 4 | 153,190,400 | 0.9% |
| Q8_0 | 106 | 80,773,120 | 0.5% |
| F32 | 360 | 10,686,464 | 0.1% |

Q4_K is only a quarter of the bytes. IQ4_XS (non-linear 4-bit with a lookup table) is as large a share as Q4_K.
`general.file_type = 15` (llama.cpp prints "Q4_K - Medium") describes the recipe name, not the contents. The
"UD" prefix means Unsloth Dynamic: per-tensor types chosen by importance matrix (`quantize.imatrix.*` keys in
`tests/fixtures/gguf_metadata.json`). **Phase 2 needs dequant/dot kernels for Q4_K, Q5_K, Q6_K, Q3_K, Q8_0,
IQ4_XS, IQ4_NL and IQ3_S, or a repack step that converts everything to one format (and then the parity
target changes).** Even within one layer the types are mixed, e.g. layer 0: attn_qkv Q5_K, ffn_down/gate IQ4_XS,
ffn_up Q3_K, ssm_alpha/beta Q8_0.

## 2. There is no plain `Q4_K_M` file; the name is `Qwen3.8-27B-UD-Q4_K_M.gguf`

The task and ARCHITECTURE.md say "the Q4_K_M file (~17 GB)". `docs/data/gguf_repo_files.txt` lists no
`Qwen3.8-27B-Q4_K_M.gguf`; the only Q4_K_M is `Qwen3.8-27B-UD-Q4_K_M.gguf`, **16,464,440,224 bytes = 15.33 GiB
(16.46 GB)**. The Unsloth guide's own run example uses `UD-Q4_K_XL` (17,559,178,144 bytes), not Q4_K_M.

## 3. The MTP head IS inside the main GGUF, as a 65th block named `blk.64`

ARCHITECTURE.md lists "whether MTP tensors are in the GGUF at all" as unknown; the repo also ships a separate
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, which suggests it is not. Payload:

- Main GGUF: `qwen35.block_count = 65` (not 64), `qwen35.nextn_predict_layers = 1`. Layer 64 has the 11 tensors of
  a full-attention block plus `blk.64.nextn.eh_proj.weight` [5120, 10240] Q6_K, `blk.64.nextn.enorm.weight`,
  `blk.64.nextn.hnorm.weight`, `blk.64.nextn.shared_head_norm.weight` (all F32 [5120]). 15 tensors, 351,008,768 bytes
  = 334.75 MiB, contiguous at the end of the file. llama.cpp reports `n_layer = 64, n_layer_all = 65`.
- HF safetensors: the same 15 tensors live at top level as `mtp.fc.weight`, `mtp.layers.0.*`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight` (not under `model.`). Mapping by shape:
  `mtp.fc` -> `nextn.eh_proj`, `pre_fc_norm_embedding` -> `nextn.enorm`, `pre_fc_norm_hidden` -> `nextn.hnorm`,
  `mtp.norm` -> `nextn.shared_head_norm`, `mtp.layers.0.*` -> `blk.64.attn_*/ffn_*`.
- The separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 bytes; header captured in
  `tests/fixtures/gguf_index.mtp.json`) is a standalone 18-tensor draft model: its own `token_embd.weight` and
  `output.weight` (Q3_K, 521 MiB each), `output_norm`, and a second copy of `blk.64.*` at different types
  (Q6_K/Q4_K). Its `general.file_type = 14` (Q4_K_S); the "Q4_0" in the filename is not what is inside.
- Config: HF `config.json` has `mtp_num_hidden_layers` only inside `text_config`; the GGUF repo's `config.json` has
  it at BOTH top level and inside `text_config`, plus `unsloth_fixed: true, unsloth_fixed_mtp: true`.
- No MTP embedding or lm_head anywhere (`mtp_use_dedicated_embeddings: false`): the draft layer must reuse
  `token_embd` / `output`.

## 4. Full-attention layers are 3, 7, 11, ... 63 (0-based), the last of each group of four

ARCHITECTURE.md: "`full_attention_interval: 4` -> 16 full-attention layers" (positions unstated). The natural
reading "every 4th starting at 0" is wrong. `config.json:text_config.layer_types` is an explicit 64-entry list
(`tests/fixtures/config.json`); full attention is at index i where `(i+1) % 4 == 0`. The GGUF tensor names
agree (`blk.3`, `blk.7`, ... have `attn_q/k/v`; `blk.0,1,2` have `ssm_*`). Model card: "3 x (Gated DeltaNet ->
FFN) -> 1 x (Gated Attention -> FFN)". Both HF and GGUF layer indices are 0-based and identical.

## 5. Vision tower: 333 tensors in safetensors, zero in the GGUF

`st_index.json` has 333 `model.visual.*` tensors (27 blocks, patch embed, pos embed, merger). The main GGUF has
none; vision ships as separate `mmproj-BF16.gguf` / `mmproj-F16.gguf` (about 930 MB each). `vision_config.model_type`
is `qwen3_5` in the HF config but `qwen3_5_vision` in the GGUF repo's config.

## 6. Embeddings are not tied, and the two matrices are quantized differently

`tie_word_embeddings: false` (both levels). GGUF: `output.weight` Q6_K 994.63 MiB and `token_embd.weight` Q4_K
682.03 MiB, separate tensors. The pinned non-layer set is 1,758,126,080 bytes (1.68 GiB) plus the 334.75 MiB MTP
block, about 2.0 GiB before any layer is resident.

## 7. Layers are contiguous on disk, so `pack.rs` is not needed

ARCHITECTURE.md left "whether GGUF layer tensors are contiguous on disk" open and reserved a `pack` subcommand.
`docs/data/gguf_layout.txt`: all 65 layers contiguous, zero gap bytes, no foreign tensors inside any span,
tensors adjacent in file order, alignment 32. Largest real layer: 63 (282,298,368 bytes = 269.2 MiB); largest
block overall: 64 / MTP (334.75 MiB, pinned, not streamed); smallest: layer 14 (172.1 MiB). Layer sizes vary
by 60% because of the per-tensor type choice, so the ring slot must be sized to layer 63, not to an average.
Tensor data starts at byte 10,996,640 (the 11 MB header is mostly the 248,320-entry vocab and merges).

## 8. EOS, BOS and PAD disagree between files

| file | bos | eos | pad |
|---|---|---|---|
| `config.json:text_config` | 248044 | 248044 | null |
| `generation_config.json` | 248044 | [248046, 248044] | 248044 |
| `tokenizer_config.json` | none (`add_bos_token: false`) | `<\|im_end\|>` = 248046 | `<\|endoftext\|>` = 248044 |
| GGUF `tokenizer.ggml.*` | 248044 | 248046 | 248055 = `<\|vision_pad\|>` |
| GGUF repo `config.json` (top level) | | | 248055 |

The engine must stop on both 248046 and 248044. The GGUF has no `tokenizer.ggml.add_bos_token` key at all.

## 9. The GGUF's embedded chat template is not the HF chat template

`docs/data/gguf_vs_hf_tokenizer.txt`: GGUF `tokenizer.chat_template` is 9,993 chars vs 8,952 in
`chat_template.jinja`. Unsloth's changes: a `developer` role, merging of multiple system messages,
`reasoning_effort: high` aliased to `xhigh`, tool-call argument validation, and removal of the "No user query
found" exception. The HF file is the reference for Phase 5; the GGUF copy is not.

## 10. Thinking-off is a prompt prefill, not a flag

Model card and ARCHITECTURE.md: "thinking can be disabled per request". `docs/chat-template.md`: with
`enable_thinking=False` the template appends `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n`, i.e. an
empty think block is inserted into the prompt; with thinking on it appends `<think>\n` and, for `xhigh`
(default) or `low`, injects a reasoning-effort sentence into the system message. `<think>` (248068) and
`</think>` (248069) are `special: false` in `tokenizer_config.json` but the HF tokenizer still maps the literal
text to those single ids (`tok_cases.json`, case `special_lookalike`).

## 11. Vocab is padded: 248,320 rows, 248,077 real ids

Model card says "248,320 (Padded)". `tokenizer.json` has 248,044 vocab entries plus 33 added tokens (max id
248,076). The GGUF fills ids 248,077..248,319 with `[PAD248077]`.. and `token_type = 5` (unused). The 27 control
tokens have type 3 and the 6 `special: false` added tokens (`<tool_call>`, `<think>`, ...) have type 4.

## 12. GGUF hyperparameter names are Mamba vocabulary, not DeltaNet vocabulary

| GGUF key (`tests/fixtures/gguf_metadata.json`) | value | config.json field it actually corresponds to |
|---|---|---|
| `qwen35.ssm.state_size` | 128 | `linear_key_head_dim` / `linear_value_head_dim` |
| `qwen35.ssm.group_count` | 16 | `linear_num_key_heads` |
| `qwen35.ssm.time_step_rank` | 48 | `linear_num_value_heads` |
| `qwen35.ssm.inner_size` | 6144 | `linear_num_value_heads` x `linear_value_head_dim` |
| `qwen35.ssm.conv_kernel` | 4 | `linear_conv_kernel_dim` |
| `qwen35.rope.dimension_count` | 64 | `head_dim` x `partial_rotary_factor` |
| `qwen35.rope.dimension_sections` | [11, 11, 10, 0] | `rope_parameters.mrope_section` = [11, 11, 10] (GGUF has a 4th entry) |
| `qwen35.attention.key_length` / `value_length` | 256 | `head_dim` |

## 13. Tensor naming differs everywhere, and one tensor changes rank

From the size-matched side-by-side in `docs/data/gguf_layout.txt` (not an assumed map):

| HF safetensors | GGUF | note |
|---|---|---|
| `model.language_model.layers.L.input_layernorm` | `blk.L.attn_norm` | |
| `...post_attention_layernorm` | `blk.L.post_attention_norm` | |
| `...linear_attn.in_proj_qkv` [10240,5120] | `blk.L.attn_qkv` | DeltaNet tensor under an `attn_` prefix |
| `...linear_attn.in_proj_z` [6144,5120] | `blk.L.attn_gate` | |
| `...linear_attn.in_proj_a` [48,5120] | `blk.L.ssm_alpha` | Q8_0 |
| `...linear_attn.in_proj_b` [48,5120] | `blk.L.ssm_beta` | Q8_0 |
| `...linear_attn.A_log` [48] | `blk.L.ssm_a` | F32 |
| `...linear_attn.dt_bias` [48] | `blk.L.ssm_dt.bias` | F32 |
| `...linear_attn.conv1d.weight` [10240,1,4] | `blk.L.ssm_conv1d.weight` [10240,4] | rank 3 -> rank 2 |
| `...linear_attn.norm` [128] | `blk.L.ssm_norm` | |
| `...linear_attn.out_proj` [5120,6144] | `blk.L.ssm_out` | |
| `...self_attn.q_proj` [12288,5120] | `blk.L.attn_q` | 2 x 24 x 256: Q and output gate fused (`attn_output_gate: true`) |
| `...self_attn.k_proj`, `v_proj` [1024,5120] | `blk.L.attn_k`, `attn_v` | |
| `...self_attn.o_proj` [5120,6144] | `blk.L.attn_output` | |
| `...self_attn.q_norm`, `k_norm` [256] | `blk.L.attn_q_norm`, `attn_k_norm` | |
| `model.language_model.embed_tokens` | `token_embd` | |
| `model.language_model.norm` | `output_norm` | |
| `lm_head` | `output` | |

The model card's "Number of Attention Heads: 24 for Q" describes 6144 output features; the actual `q_proj` has
12288 because of the attention output gate, which the card does not mention.

## 14. RoPE is partial and multimodal-shaped even for text

Model card: "Rotary Position Embedding Dimension: 64". Config expresses it as `partial_rotary_factor: 0.25` of
`head_dim: 256`, `rope_theta: 10,000,000`, `rope_type: default`, `mrope_section: [11, 11, 10]`,
`mrope_interleaved: true`. For text-only input the transformers reference feeds identical positions to the T/H/W
streams, so the interleave is a no-op and this reduces to plain RoPE on the first 64 dims (see
`docs/model-facts.md`, "inferred" table). llama.cpp reports `rope type = 40`, `rope scaling = linear`,
`freq_scale = 1`: no YaRN by default; the 1M-context recipe in the model card is an explicit override. Only the
16 full-attention layers have any rotary inputs; DeltaNet layers have no position tensors or parameters.

## 15. Model naming: "Qwen3.8" is a product name, the architecture is `qwen3_5`

`config.json`: `architectures = ["Qwen3_5ForConditionalGeneration"]`, `model_type = qwen3_5`,
`text_config.model_type = qwen3_5_text`. GGUF: `general.architecture = qwen35`, `tokenizer.ggml.pre = qwen35`.
ARCHITECTURE.md had the class name right. The bundled llama.cpp in llama-cpp-python 0.3.35 supports `qwen35`.

## 16. Config field-name drift between the two config.json copies

| HF `config.json` | GGUF repo `config.json` |
|---|---|
| `text_config.dtype: bfloat16` | `text_config.torch_dtype: bfloat16` |
| `transformers_version: 5.8.0.dev0` | `transformers_version: 5.16.0.dev0` |
| no top-level `pad_token_id` | `pad_token_id: 248055` |
| no top-level `mtp_num_hidden_layers` | `mtp_num_hidden_layers: 1` (duplicated) |
| `vision_config.model_type: qwen3_5` | `qwen3_5_vision` |
| | `unsloth_fixed: true`, `unsloth_fixed_mtp: true` |

`config.json` nests text parameters under `text_config` as ARCHITECTURE.md expected; a flat reader would find
none of them.

## 17. Parameter and size counts

Model card: 27B. llama.cpp counts 27.32 B params in the GGUF (64 layers + MTP block + separate embed and output,
no vision), 4.82 bits per weight. `model.safetensors.index.json:metadata.total_size` = 55,562,855,904 bytes bf16 =
27.78 B params including the 333 vision tensors. The repo's 18 shards total 55,586,114,863 bytes = 51.77 GiB
(ARCHITECTURE.md: "bf16 approximately 56 GB").

## 18. `model.safetensors.index.json` has no shapes

The task expected "tensor names and shapes" from the index json. It only maps names to shard files. Shapes and
dtypes in `tests/fixtures/st_index.json` come from the safetensors headers of the 18 shards, fetched with HTTP
range requests by `tools/st_headers.py` (no shard data downloaded).

## 19. Tokenizer implementation details the engine must match

`tokenizer_config.json`: `tokenizer_class: Qwen2Tokenizer`, `add_bos_token: false`, `add_prefix_space: false`,
`pretokenize_regex` given verbatim (see `docs/model-facts.md`). GGUF `tokenizer.ggml.model = gpt2`,
`tokenizer.ggml.pre = qwen35`; merges are identical (247,587). llama.cpp maps GGUF `padding_token_id` to
`<|vision_pad|>`. The 45-case fixture (`tests/fixtures/tok_cases.json`) round-trips on decode in all cases.

## 20. vLLM and Unsloth serve the MTP head differently from a CPU engine

vLLM recipe: `--speculative-config '{"method":"mtp"|"qwen3_5_mtp","num_speculative_tokens":3}'`, plus a DFlash2
drafter (`incoai/Qwen3.8-27B-DFlash2`, 7 draft tokens). Unsloth's llama.cpp path reads the MTP layer from the
main GGUF (the guide says "MTP enabled ... 1-2 GB extra headroom"); the separate `MTP/` file exists for the
draft-model path. ARCHITECTURE.md's plan (draft k tokens with the in-checkpoint head, batched verify) is
consistent with the in-file layout found in item 3.

## 21. Reference timing on this machine (for planning, not a divergence)

`tests/fixtures/ref_llamacpp/summary.txt` (bartowski Q4_K_M) and `tests/fixtures/ud/ref_llamacpp/summary.txt`
(Unsloth UD): llama.cpp CPU, llama-cpp-python 0.3.35, 6 threads, i7-9750H, 32 GB. Cold load of the 16.5 GiB
bartowski file 294 s, first prompt eval 220 s (page-in), then 9 to 19 s for 4 to 39 prompt tokens; greedy decode
4.2 to 4.7 s per token at short context and 6.4 s per token after the 39-token prompt (UD file: 3.8 to 4.0 s and
9.1 s). ARCHITECTURE.md's cost model (`20 ms x GB_ram` = about 0.3 s/token for 16 GB in RAM) is more than 10x
more optimistic than llama.cpp on this CPU; the target table's "32 GB: fits fully, reference" row should be
measured, not assumed.

# Addendum (same day): second GGUF source, MTP identity

## 22. Qwen publishes no GGUF; bartowski's Q4_K_M has 6 ggml types, Unsloth's has 9

`docs/data/qwen_gguf_repo_files.txt`: `Qwen/Qwen3.8-27B-GGUF` does not exist publicly (401/404 anonymously; the
Qwen author search returns only `Qwen3.8-27B` and `Qwen3.8-27B-FP8`). `docs/data/q4km_type_breakdown.txt`, from
headers only:

| file | bytes | distinct types | composition |
|---|---|---|---|
| `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | 9 | Q5_K 29.4%, IQ4_XS 28.9%, Q4_K 25.6%, Q6_K 11.0%, IQ4_NL, Q3_K, IQ3_S, Q8_0, F32 |
| `bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` | 17,772,537,440 | 6 | Q4_K 58.1%, Q6_K 33.2%, Q8_0 4.5%, Q5_K 2.3%, Q4_0 1.3%, F32 0.6% |

bartowski's file is the classic llama.cpp Q4_K_M recipe (no IQ types): attn_qkv Q6_K, ffn_gate/up Q4_K,
ffn_down half Q6_K half Q4_K, attn_v/attn_q half Q6_K half Q4_K, ssm_out half Q8_0 half Q4_K, ssm_alpha/beta/A/dt
F32, and the whole MTP block (`blk.64.*` quantised tensors) in Q4_0. It is 1.3 GB larger than the UD file. It was
chosen as the primary GGUF for Phase 1+ because it needs the fewest kernels (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0); the UD
fixtures are kept under `tests/fixtures/ud/` and `docs/data/ud/`. Both files have `general.architecture = qwen35`,
`block_count = 65` and `blk.64.nextn.*`.

Layout of the bartowski file (`docs/data/gguf_layout.txt`, now the primary): 866 tensors, all 65 blocks contiguous
with zero gap bytes, data at byte 10,995,296. Because the recipe is uniform the layer sizes sit in a narrow band:
largest layer 0 at 269,681,536 bytes (257.2 MiB), smallest layer 11 at 214,018,048 bytes (204.1 MiB), versus 172 to
269 MiB in the UD file. The pinned set is unchanged (output Q6_K + token_embd Q4_K + norm = 1,676.7 MiB) plus a
253.7 MiB MTP block (Q4_0). Ring slot for this file: 257.2 MiB.

Metadata differences between the two GGUFs (`docs/data/gguf_metadata_summary.txt` vs `docs/data/ud/...`):

| key | bartowski | Unsloth UD |
|---|---|---|
| `tokenizer.chat_template` | byte-identical to HF `chat_template.jinja` (8,952 chars) | Unsloth-modified (9,993 chars), see item 9 |
| `tokenizer.ggml.add_bos_token` | present, `false` | absent |
| `tokenizer.ggml.padding_token_id` | 248044 `<\|endoftext\|>` (matches `tokenizer_config.json`) | 248055 `<\|vision_pad\|>` |
| `general.name` | `Qwen3.8 27B` | `Qwen3.8-27B` |
| `quantize.imatrix.*` | bartowski calibration-v6, 582 chunks | Unsloth imatrix, 1251 chunks |
| vocab, merges, token types | identical to HF (243 `[PADn]` filler ids, types 3/4/5 as in item 11) | same |

So the earlier items 8 and 9 (pad = vision pad, modified template, missing add_bos key) are Unsloth-specific; with
bartowski's file the GGUF and HF tokenizer metadata agree except for the padded vocab rows.

## 23. The separate MTP GGUF holds the same weights as `blk.64` of the main file

`docs/data/ud/mtp_blk64_diff.txt` (`tools/mtp_blk64_diff.py`; 38 MB of the MTP file fetched by range request):
the 15 `blk.64.*` tensors have identical names and shapes in both files; the 7 F32 tensors are bit-identical;
the quantised tensors dequantize to cosine 0.9998 (attn_k, attn_v: Q8_0 vs Q6_K, 2% relative RMS) and 0.9973
(nextn.eh_proj: Q6_K vs Q4_K, 7% relative RMS), i.e. the same bf16 source at different quantisation types. The
separate file is a convenience for llama.cpp's draft-model path, not extra weights. A CPU engine should read the
MTP block from the main file and ignore `MTP/`.

The same holds for bartowski's `blk.64` (`docs/data/mtp_blk64_diff.txt`, `docs/data/mtp_blk64_diff.bartowski_vs_ud.txt`):
identical names and shapes, F32 tensors bit-identical to both Unsloth files, quantised tensors at cosine 0.9953 to
0.9964 against Unsloth's copies. bartowski stores the MTP block's matrices as Q4_0 (relative RMS error about 9%
against Q8_0/Q6_K), the noisiest of the three copies; Unsloth's main file keeps them at Q6_K/Q8_0.

## 24. Two Q4_K_M recipes of the same model disagree on greedy text after four tokens

`docs/data/llamacpp_bartowski_vs_ud.txt`: same engine (llama.cpp), same prompts, bartowski Q4_K_M vs Unsloth
UD-Q4_K_M. Argmax matches on all three prompts and the top-10 overlaps 9 to 10 of 10, but raw logits differ by up
to 0.95 (mean 0.09 to 0.17) and the 39-token prompt's greedy continuation diverges at token 5 ("rivers plunge
over cliffs" vs "glaciers once shaped the landscape"). "Q4_K_M" is therefore not one model: ARCHITECTURE.md's
"Same tokens at every budget" contract can only be stated per GGUF file, and the parity target must name the
file (bartowski's, from this addendum on).

# Phase 1 findings (reading the model from the GGUF alone)

## 25. Five values the engine needs exist in config.json but have no GGUF metadata key

`ModelConfig` (crates/core/src/config.rs) is built only from GGUF KV metadata and tensor shapes. These
config.json values could not be sourced from the GGUF and were therefore NOT put into the contract
(`docs/config-mapping.md`, last table):

| HF path | value | GGUF | consequence |
|---|---|---|---|
| `text_config.hidden_act` | `silu` | no key | llama.cpp hardcodes SiLU for the `qwen35` architecture |
| `text_config.output_gate_type` | `swish` | no key | DeltaNet output gate activation is implied by the architecture |
| `text_config.mamba_ssm_dtype` | `float32` | no key | recurrent-state precision is an implementation choice |
| `text_config.rope_parameters.mrope_interleaved` | `true` | no key (`rope.dimension_sections` exists, the interleave flag does not) | irrelevant for text-only positions |
| `text_config.rope_parameters.rope_type` / scaling | `default` | no `rope.scaling.*` keys | llama.cpp assumes freq_scale 1 when absent |

Open question for the user: carry these as architecture-implied constants keyed on `general.architecture == qwen35`
(what llama.cpp does), or refuse to run without them. Phase 2 cannot start the MLP or DeltaNet gate without a decision.

## 26. Smaller Phase 1 findings

- The GGUF names one EOS (`tokenizer.ggml.eos_token_id = 248046`, `<|im_end|>`); `generation_config.json` stops on
  `[248046, 248044]`. 248044 (`<|endoftext|>`) is the GGUF's bos and pad id, so a GGUF-only engine would not stop on
  it unless it treats bos as a stop token. There is no `tokenizer.ggml.eot_token_id` key in either GGUF.
- `general.alignment` is absent from both GGUFs; the reader applies the GGUF specification's 32 (a format rule,
  documented in `docs/config-mapping.md`), not a model default.
- `text_config.eos_token_id` (248044) in config.json is not the conversational EOS; `tokenizer_config.json`'s
  `eos_token` (`<|im_end|>` = 248046) is. `tokenizer_config.json` has no BOS token at all, while the GGUF names
  248044 as bos with `add_bos_token = false`.
- The GGUF's `{arch}.rope.dimension_sections` has four entries `[11, 11, 10, 0]` versus three in config.json;
  the trailing 0 is a fourth mRoPE axis llama.cpp reserves. `2 * (11 + 11 + 10) = 64 = rope.dimension_count`.
- The `tokenizers` Rust crate reproduces the HF Python tokenizer on all 45 cases and 3 prompts with the
  `fancy-regex` backend (no C dependency); the same crate version is what the Python fixtures used (0.23.1).

# Phase 2a findings (kernels and the tiny oracle)

## 27. The GGUF DeltaNet tensors are re-tiled and transformed by the converter; HF layout is not the on-disk layout

llama.cpp `conversion/qwen.py` (commit 67a17c1, `_LinearAttentionVReorderBase`, `Qwen3NextModel.modify_tensors`):
V heads move from HF's grouped order (head `k*3+v`) to tiled order (position `v*16+k`) in the V rows of
`attn_qkv`, in `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, the V channels of `ssm_conv1d`,
and the input columns of `ssm_out`; `ssm_a` stores `-exp(A_log)`, not `A_log`; every `*norm.weight` gets `+1`
except `linear_attn.norm.weight` (the gated norm is ones-initialised, the others are zero-centered);
`conv1d.weight` is squeezed from `[C, 1, 4]` to `[C, 4]`. Attention tensors are untouched (Qwen2 path: no q/k
permutation), so `attn_q` keeps HF's per-head `[q | gate]` packing. Full table in `docs/deltanet.md`.
Engine consequence: V head `p` reads K head `p % 16`; nothing in the engine reproduces HF order.

## 28. HF facts captured for the kernels (docs/rope.md, docs/deltanet.md)

- RoPE rotates only the first 64 of 256 head dims, rotate-half pairing `(i, i+32)`, after q/k RMSNorm; for
  text the three mRoPE streams are equal, so `mrope_interleaved` is a no-op. The converter warns
  "Unknown RoPE type: default" and writes no scaling keys.
- DeltaNet op order is decay, read, delta write, read, with `l2norm(q)`, `l2norm(k)` (eps 1e-6) and
  `q *= 1/sqrt(128)` before the step; `g = ssm_a * softplus(a + dt_bias)`, `beta = sigmoid(b)`; the output norm
  is `weight * rmsnorm(out)` then `* silu(z)`.
- HF's `DynamicCache` keeps 4 conv samples per channel but only the last 3 feed the next step; our state keeps 3.
- HF prefill uses the chunked delta rule; its output differs from the sequential recurrence by at most
  4.6e-7 on outputs of magnitude 0.96 (`tests/fixtures/layers/layers/manifest.json`, `chunk_vs_seq_max_diff`),
  so a sequential prefill matches within the derived budgets.

## 29. Tiny oracle: converter path succeeded with two forced deviations

`tools/make_tiny_checkpoint.py` -> `models/tiny/tiny-f32.gguf` (138 tensors, 133 MB) via the real
`convert_hf_to_gguf.py` (Qwen3_5ForCausalLM path). The converter hashes the tokenizer to choose
`tokenizer.ggml.pre` and asserts the vocab fits `vocab_size`, so the tiny model reuses the real tokenizer
and the real 248,320 vocab instead of 512; and it asserts on `mtp_num_hidden_layers = 1` without `mtp.*`
tensors, so a random MTP block is included (it becomes `blk.9.*`, ignored by the oracle). Everything else
is as specified: 9 layers, interval 4, hidden 64, 2 KV heads, DeltaNet 2 K x 6 V heads of 16.
Result: 60/60 greedy tokens, per-layer hidden states at 1% of budget, logits within 3.6e-7 of HF
(`docs/data/phase2a_tests.txt`).

## 30. gguf-py cannot quantise K-quants

`gguf.quants.quantize` implements only Q4_0 and Q8_0 (plus F32/F16 pass-through). K-quant dot-kernel
fixtures therefore use rows sliced from the real bartowski GGUF plus hostile blocks tiled from the Phase 1
dequant fixtures with rewritten f16 scales (`tools/fixtures/gen_kernels.py`).

# Phase 2b findings (real layers, noise floor, AVX2)

## 31. Against the dequantised-GGUF reference, an f32-activation engine sits at 0.3% of the derived budget; the Q8 engine path is 2^15 times further away, by construction

`tests/fixtures/ref_gguf/` is HF transformers in float32 on the bartowski GGUF dequantised with gguf-py
(`tools/ref_forward.py --weights gguf`; 66 MB, 64 layers x 3 prompts, every position). The engine was run
against it in two modes (`crates/core/tests/real_layers.rs`, `docs/data/phase2b_parity_0_63.log`):

| mode | what it computes | layers 0..63, worst max\|diff\| / budget | logits vs ref_gguf (3 prompts) |
|---|---|---|---|
| f32 | weights dequantised in Rust (bit-identical to gguf-py), f32 dot | 0.0027 (layer 0, sentence: 1.1e-5 vs 4.2e-3); never above 0.003 | max\|diff\| 1.2e-5 to 2.1e-5 (budget 0.27 to 0.33), argmax 3/3, top-10 10/10, argmax at every prompt position 48/48 |
| q8 | the engine's real path: quantised rows x Q8_0 activations | relative max error 3e-4 to 1.9e-2 per layer (worst layer 55, sentence) | max\|diff\| 0.11 to 0.16, argmax 3/3, top-10 9 to 10 of 10, argmax at every position 48/48 |

Rule 5's budget (`k * sqrt(width) * f32_eps * max|h|`, k = 16 per layer, compounding as `k * (L + 1)`) can
only gate the f32 mode: quantising an activation block to 8 bits moves each element by up to `max|block| / 254`,
about 2^-8 relative, against 2^-23 for f32 rounding. So the parity gate is: f32 mode within budget (checks every
binding, op order, re-tiling, norm, RoPE, DeltaNet and cache), q8 mode measured and gated on argmax / top-10.
The tiny oracle never saw this because its GGUF is F32 (the f32 dot path); the Q8 kernels themselves are checked
bit-exactly against dequantised-Q8 fixtures in `tests/kernels.rs`.

The q8 mode is also compared with llama.cpp on the same file (`tests/fixtures/ref_llamacpp`): max|diff| 0.32 to
0.39, argmax 3/3, top-10 9 to 10 of 10. llama.cpp is itself 0.31 to 0.38 from the f32 reference (it quantises
activations to Q8_K, 256-value blocks with one scale; ours are Q8_0, 32-value blocks), so the two 8-bit engines
disagree with each other about as much as either disagrees with float32, and ours is nearer to float32
(0.11 to 0.16). "Matching llama.cpp" is therefore a loose target: two correct Q8 engines differ by ~0.35 in raw
logits on these prompts; a wrong binding differs by tens.

## 32. `-exp(A_log)` cannot be inverted bit-exactly in float32 for a third of the DeltaNet heads

The converter stores `ssm_a = -exp(A_log)` in float32. Recovering `A_log = log(-ssm_a)` and letting HF compute
`-exp(A_log)` again reproduces the stored value on all 48 heads in only 17 of the 48 DeltaNet layers
(`tests/fixtures/ref_gguf/info.json`, `a_log_exact_per_layer`); on the other layers some heads have no float32
neighbour whose `exp` lands on the stored value, and the nearest one is 1 ulp off. The engine uses `ssm_a` as
stored, so it is exact; the reference carries a 6e-8 relative error on those decay exponents, invisible at the
budget (finding 31's f32 numbers include it). A reference built through `A_log` cannot be bit-exact on the gate;
one built on the GGUF's `ssm_a` directly could be.

## 33. The decode step and the prefill are the same code today, so "incremental == prefill" is bit-exact by construction

`GatedDeltaNet::forward_prefill` and `GqaAttention::forward_prefill` are T sequential `forward_token` calls
(Phase 2a). The 2b check (prefill T-1 positions, clone the state, decode the last token) therefore passes bit for
bit on all 64 layers without testing anything new. It stays in the harness so that a batched or chunked prefill
(Phase 3) has to keep it true; HF's own chunked prefill differs from its sequential recurrence by 5e-7 (finding 28).

## 34. Timing facts for planning: everything on this box streams from a SATA disk

`models/` lives on D:, the 1 TB SATA HDD (ST1000LM049), not the NVMe. Consequences measured this session:
the 66 MB reference dump took 1206 s (64 layers x 3 prompts; gguf-py dequantisation 8 to 10 s per layer plus
the read); the bf16 reference took 45 to 70 s per layer, almost all of it reading 0.87 GB of shards from disk;
the Rust parity harness streams the GGUF once per run and took 3012 s for 0..63 in both modes with scalar
kernels (45 s per layer over 48 prompt tokens, 95 s per lm_head pass); with AVX2 the q8 mode drops from 19 s to
5 s per layer (`docs/data/phase2b_parity_0_7_avx2.log`). The reference and the engine cannot both fit next to
the user's other processes (about 22 GB of the 32 GB were taken), which is why both stream one layer at a time.

## 35. Quantisation noise floor: the published bf16 model and the Q4_K_M file differ by 0.5 to 0.7 in raw logits; the two Q8 engines differ by the same amount from each other

`docs/data/quant_noise_floor.txt` (`tools/noise_floor.py`): HF in float32 on the bf16 shards versus HF in
float32 on the dequantised bartowski GGUF, identical code, only the weight bytes differ.

| | layer 0 | layer 7 | layer 31 | layer 63 | final norm | logits |
|---|---|---|---|---|---|---|
| max\|diff\| / max\|bf16\| | 3.5e-3 to 1.2e-2 | 1.3e-2 to 2.6e-2 | 5e-3 to 1e-2 | 3.8e-2 to 4.9e-2 | 2.2e-2 to 3.8e-2 | raw max\|diff\| 0.52 to 0.68, mean 0.07 to 0.11 |
| rms(diff) / rms(bf16) | 1.0e-2 to 1.3e-2 | 1.7e-2 to 2.2e-2 | 3.4e-2 to 5.2e-2 | 4.9e-2 to 6.4e-2 | 5.1e-2 to 6.5e-2 | argmax 3/3, top-10 9 to 10 of 10 |
| cosine | 0.99995 | 0.9998 to 0.9999 | 0.9987 to 0.9994 | 0.9980 to 0.9988 | | |

So Q4_K_M keeps the residual stream within about 5% RMS of the bf16 model after 64 layers and moves raw logits
by up to 0.7 on these prompts, without changing the argmax. Against bf16, llama.cpp sits at 0.63 to 0.66
(weight noise plus its Q8_K activations) and our engine's q8 mode, by finding 31, at roughly the same distance.
This is the floor the ladder's "same tokens at every budget" claim lives above: it is a statement about one
GGUF file, and any engine reading that file is 0.5 to 0.7 from the bf16 model before its own rounding enters.
Note also that the bf16 run died silently in its lm_head pass on Windows (safetensors slicing of the 2.5 GB
tensor); the logits were recomputed from the saved final-norm dumps by `tools/ref_logits_from_dump.py`, and the
shard store now loads that tensor whole.

## 36. AVX2 kernels are bit-identical to scalar, and compute-bound rather than memory-bound

`docs/simd.md`: the five dot kernels, the Q8_0 quantiser, `rmsnorm` and `softmax` have AVX2 versions selected
at runtime; each produces the same bits as the scalar kernel (`tests/avx2.rs`: 93 fixture rows, 5000 random
rows, 1000 quantiser rows including 167 with exact .5 ties, max diff 0), and layers 0..7 of the real model give
identical numbers with the scalar path forced (`docs/data/phase2b_parity_0_7_scalar.log` vs `_avx2.log`). To
get there the scalar `inv_rms` and softmax sums were redefined as eight interleaved lanes with a fixed reduction
order; the fixture tests and the tiny oracle (60/60) still pass. `docs/data/kernels_bench.txt` (17408 x 5120
matrices, best of three runs): single thread AVX2 is 2.9x (Q4_K) to 13x (Q6_K) the scalar speed; with all
threads the dot kernels reach 2.9 (Q4_K) to 6.1 (Q8_0) GB/s of weights, a fifth of this laptop's memory
bandwidth: the per-block horizontal reductions and the serial f32 accumulation kept for bit-identity bound the
kernels, not DRAM. At the file's mix (58% Q4_K, 33% Q6_K) that is about 3.4 GB/s, i.e. 5 s per token for the
17.7 GB file, llama.cpp's neighbourhood on this CPU (4.2 to 4.7 s, finding 21). The three bench runs disagree
by up to 2x on the same kernel (an idle-machine run was the slowest), which on a laptop after hours of full
load reads as thermal throttling; the ladder numbers in Phase 4 must be taken on a cooled machine with the
clock frequency logged. Doubling the kernel throughput needs a batched reduction (four blocks per horizontal
add) and is the first Phase 3 speed item.
