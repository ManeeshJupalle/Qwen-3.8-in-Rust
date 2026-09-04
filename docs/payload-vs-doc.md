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

**Phase 3 update (2026-09-03):** the bartowski GGUF was moved off the HDD to the SSD at
`C:\models\Qwen3.8-27B-Q4_K_M.gguf` (C: is the NVMe, D: the SATA HDD); `models/Qwen3.8-27B-GGUF-bartowski/`
is now empty. Every path that named it follows: `crates/core/tests/common/mod.rs::gguf_path` (constant
`DEFAULT_GGUF`, overridable with `AQUEDUCT_GGUF`), `tools/ref_forward.py` (`GGUF_DEFAULT`),
`tools/fixtures/gen_kernels.py` (`GGUF_PATH`) and the CLI default for `aqueduct run`. The bf16 shards,
the tiny oracle and the llama.cpp checkout stay under `models/` on D:. Phase 3 load times are therefore SSD
numbers (`docs/data/phase3_timing.txt`) and not comparable with the disk timings above.

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
# Phase 3 findings (kernel throughput, resident model, batched prefill)

## 37. ggml's Q8_K activations break the frozen Phase 2b ceilings; the engine keeps Q8_0-grain activations under ggml's kernel structure

Phase 3.1 rebuilt the K-quant dot path the way ggml's AVX2 kernels do it (`docs/kquant-dot.md`): exact integer
inner loops (`maddubs` / `madd`), 8-lane f32 accumulation across the whole row, one horizontal reduction per row,
the sub-block mins folded through the activation block sums, software prefetch. With ggml's Q8_K activation
format (256 values, one f32 scale, `bsums`) the q8 mode's relative error after layer 0 on the `capital` prompt is
2.1e-3 against the ceiling of 1.7e-3 (2 x the Phase 2b value of 8.7e-4): one scale per 256 values is about 2.5 x
noisier than one per 32, which is exactly the gap finding 31 measured between llama.cpp (0.31 to 0.38 from the
f32 reference) and our Q8_0 path (0.11 to 0.16). Rule 5 is a hard rule, so the production path keeps Q8_0-grain
activations (per-32 f16 scales, plus the derived `d32`, `dxs`, `off6` the kernels read) inside the same lane
structure: per 32-element sub-block one f32 factor `d_x * (d * sc)` scales the exact lane sums, the mins go
through `(m * -dmin) * (d_x * sum q_x)`, Q6_K's `-32` is subtracted in integer from two lanes. Layers 0..7 then
sit at most at 0.63 of their ceilings (layer 3, `sentence`). Q8_K stays available as an opt-in
(`aqueduct run --q8k`, `bench kernels` reports both) and its kernels and quantiser are tested bit-exactly
against a Python port of `quantize_row_q8_K_ref` (gguf-py has no Q8_K).

## 38. A single thread streaming weights from DRAM was latency-bound, not compute-bound, until software prefetch

The rebuilt Q4_K kernel ran at 6.6 GB/s single-threaded from L2 but 3.3 GB/s from DRAM on a 14.7 MB matrix,
while the sum-reduce ceiling for one thread is 14 to 16 GB/s (`docs/data/membw.txt`): the hardware prefetcher
does not keep a single 2.9 KB-row stream ahead of the integer kernel. Prefetching 1 KB ahead (one `prefetcht0`
per 64 bytes of row, 3 to 4 per super-block) lifted the DRAM number to the cache number (Q4_K 6.7 to 7.4 GB/s,
Q6_K 8.7 GB/s single-threaded) and the all-threads numbers from 12 to 15 GB/s to 20 to 26 GB/s (70 to 90 % of
the 28.9 GB/s ceiling) on the same matrix. ggml does not prefetch; it gets its outstanding misses from many
threads. Single-thread Q4_K / Q5_K remain at 6.4 to 7.4 GB/s against the 8 GB/s gate target: the remaining gap
is instruction count (about 115 uops per 144-byte super-block), not memory.

## 39. The lane-structured kernels are bounded by the term magnitudes, not by the result: the Phase 2 budget model does not apply

Each of the 8 f32 lanes (and the 8 min lanes) is a partial sum over elements of random sign and cancels against
the others only in the final reduction, so its rounding error is relative to the lane, not to the result. On the
dot fixtures the sum of term magnitudes is 40 to 90,000 times the result; a hostile Q5_K row lands 28 ulp of
its result from the exact value while being at 1.4 % of `k * eps * sum_j (|d sc q_j| + |dmin m|) |x_j|`, the
bound the tests now use for K-quant rows (`tests/common/mod.rs::kquant_terms_budget`). The Phase 2 budget
`k * sqrt(n) * eps * max|term|` failed such rows by 5 x. ggml has the same property; the whole-layer effect is
what the frozen ceilings gate, and layers 0..7 show no visible change (0.63 of ceiling worst).

## 40. The laptop throttles compute by up to 2.5 x within an hour while DRAM bandwidth does not move

The same AVX2 Q4_K matvec measured 26 GB/s (6 threads) early in the session and 9 to 18 GB/s after scalar
benches and test suites had run for an hour; cache-resident (compute-only) speed swung from 9.1 to 3.6 GB/s for
Q6_K; `membw` stayed at 29 to 30 GB/s throughout; `typeperf "% Processor Performance"` read 118 % of the 2.6 GHz
nominal (about 3.07 GHz all-core) under a 6-thread AVX2 load. Consequences: every "% of membw" number in this
phase is filed as a back-to-back pair (membw, kernels, membw) in one thermal state
(`docs/data/kernels_bench.txt`, "consistent-state pair"), and the honest reading of the kernels is the
best-of-session number with the pair as the floor. Phase 4 ladder numbers need a cooled machine and the clock
counter logged next to them.
## 41. The batched prefill is bit-identical to the token-by-token feed, by construction, on the real model

`Model::prefill` (Phase 3.3) runs every projection of a layer as one `matmul` over the prompt (each weight row
read once, applied to all `T` activations while it sits in L1), the conv window and the DeltaNet recurrence as
`T` sequential steps per head, and causal attention of all `T` queries over the freshly filled cache in one
pass. Because `matmul` computes row `r` of token `t` with the same `one_row` kernel `matvec` uses, and the
per-token parts are the same code, `tests/phase3_e2e.rs` finds max|diff| = 0 on the residual stream of every
position, on the last-position logits, on the carried DeltaNet state and conv window, and on the KV cache, for
all three prompts (finding 33 predicted this for the trivial prefill; it now holds for a real batched one).
Speed: 1.9 to 2.4 prompt tokens per second, 2.9 to 3.3 x the sequential feed. The sequential feed costs a
decode step per token (1.4 to 1.5 s); the batched form is compute-bound (48 activations per weight row), so
the next prefill win is a cache-blocked GEMM, not fewer weight reads.

## 42. Greedy decode agrees with llama.cpp for 16/16, 16/16 and 5/16 tokens; the divergence sits at a 0.07 logit margin

`tests/phase3_e2e.rs`, 32 greedy tokens per prompt from the batched prefill, first 16 against
`tests/fixtures/ref_llamacpp`: `capital` and `fib` match 16/16 (" Paris.\nThe capital of Germany is Berlin.\nThe
capital of Italy is Rome.\nThe capital of Spain is Madrid.\nThe capital of Portugal is"; the Fibonacci function
with the `elif n == 1` branch). `sentence` matches 5/16: at step 5 our top-1 (`glaciers`, id 91420) beats
llama.cpp's (`rivers`, 34343) by a top-1 / top-2 margin of 0.0708 raw logits, far inside the noise between two
Q8 engines on this file (0.29 to 0.44 max|diff|, finding 31) and our own distance from the f32 reference (0.15).
The continuation stays coherent (" In the highlands, glaciers once shaped the landscape, leaving behind U-shaped
valleys and moraines as they retreated. The interplay of erosion and deposition continues to") and every later
`sentence` step has margins of 0.05 to 7.8, so a text prompt of that length is a coin-toss chain between any two
correct engines; `--ids` with the reference logits, not token equality, is the test channel (as planned).

## 43. Fully resident on the 32 GB box: 30.6 s load from the SSD, 17.98 GB peak RSS, 1.4 to 1.6 s per token = 37 to 41 % of memory bandwidth

`Model::load` reads 17.523 GB of tensors in 30.6 s (570 MB/s; only 9 GB of RAM were free, so the OS paged
editors and the browser out while loading). Peak RSS 17.975 GB against 17.686 GB expected (weights + DeltaNet
state + KV cache for 48 + 32 positions): ratio 1.016. Decode at 12 threads: 1.562, 1.453 and 1.403 s per token
for the three prompts, i.e. 10.8 to 12.0 GB/s of weights (16.84 GB per token: all weights minus the embedding
table plus one row) = 37 to 41 % of the 28.9 GB/s ceiling, on a machine in its throttled state (finding 40),
where the matvec kernels themselves were measuring 8 to 18 GB/s. The per-token budget is therefore mostly the
kernels' throttled throughput plus the non-matvec work (48 DeltaNet steps of 48 x 128 x 128 f32, conv, norms,
the 248,320-row lm_head): with the kernels at their cool-machine 20 to 26 GB/s the same token would take about
0.8 s. llama.cpp on this CPU took 4.2 to 4.7 s per token (finding 21, HDD-resident, throttled or not unknown).
## 44. The q8 path's per-layer error on one prompt moves by up to 3 x between two correct kernel orders; the rule-5 ceilings are per layer, not per prompt

`tests/fixtures/q8_ceilings.json` freezes the Phase 2b q8-mode relative errors per layer and per prompt. The
first 0..63 run with the Phase 3 kernels (same Q8_0 activations, different summation order) failed at layer 8
on `fib`: 1.78e-3 against 2 x 6.3e-4, while the same layer's `capital` and `sentence` values (1.6e-3, 2.8e-3)
barely moved. Where a prompt's 2b number happened to land low, a 2 x margin on that single number is inside
the run-to-run noise of a Q8 path: the activation rounding is the same size, its realisation differs. Rule 5
says "per layer", so the test takes the factor times the largest of the three prompts' 2b values as the
layer's ceiling (5.6e-3 at layer 8, the failing value at 0.32 of it); the per-prompt numbers stay in the
fixture for reference, and the logits ceilings stay per prompt (last-position max|diff| vs ref_gguf, 2 x).
## 45. Greedy ids are identical at 6 and 12 threads on the real model, and identical to the gate test's run

`aqueduct run --ids <sentence> --max-tokens 16 --ids-only` at `--threads 6` and `--threads 12` (after the
64-layer parity run, machine throttled) produced the same 16 ids as `tests/phase3_e2e.rs` did at 12 threads
an hour earlier: `733,279,1496,7909,11,91420,2957,25432,279,17865,11,9101,4661,533,33582,82446`. The pool
partitions rows in contiguous chunks and every row is computed by one participant with one kernel, so the
determinism contract (same ids at every thread count) holds end to end, not only in the kernel tests.

## 46. End-of-session numbers are the throttled floor, not the kernels' capability

After about four hours of load the last consistent-state pair (`docs/data/kernels_bench.txt`) read membw 23 to
25 GB/s (29 to 30 earlier), Q4_K 15.4 GB/s at 6 threads (62 to 65 % of the paired membw), Q5_K 5.2 (21 %),
Q6_K 6.9 (28 %), single-thread 2.1 to 5.2 GB/s, and decode 2.85 s per token at both 6 and 12 threads
(`docs/data/phase3_timing.txt`), twice the 1.40 to 1.56 s measured 90 minutes earlier on the same binary. The
production (Q8_0-grain) kernels were rewritten after the machine had already heated, so they were never
measured cool; their cool-machine numbers are expected between the Q8_K-activation kernels' cool numbers
(Q4_K 26, Q5_K 22, Q6_K 26 GB/s at 6 threads, 75 to 92 % of membw) and the throttled pairs above, since they
run about 10 to 20 % more instructions per super-block and are memory-bound at all threads when cool. Q5_K
falls further than the others when hot (its DRAM number drops to a third of Q4_K's while its cache-resident
number equals Q4_K's), which points at its prefetch spacing (176-byte blocks) and is the first thing to look at
next session; it is 2.3 % of the file (`attn_k`, `attn_output`).
**Cool-down pair (added at the end of the session).** After 8 minutes idle: membw 29.7 / 30.7 GB/s at 6 threads
before / after, Q4_K (production, Q8_0 activations) 5.8 GB/s single, 20.6 at 6 threads (69 %), Q5_K 3.9 / 7.3
(24 %), Q6_K 4.1 / 7.6 (25 %), Q8_0 2.8 / 7.7; the Q8_K opt-in kernels came out slower than the production ones
in this state (Q4_K 8.2 at 6 threads) although they were faster when the box was cool, so the throttling is not
a uniform clock factor. Decode after the pair: 2.30 s per token (7.3 GB/s) at 12 threads, ids unchanged. The
production Q5_K and Q6_K kernels therefore did not demonstrate the 50 % gate at the 5120 x 5120 shape in any
state measured today (53 % and 62 % at 17408 x 5120, throttled); `docs/phase3-report.md` lists what to try.
# Phase 3.5 findings (kernel decision)

## 47. This box's performance counters report cumulative CPU time as a rate, and that invented a runaway service

Phase 3.5 first read the machine with the Windows performance-counter subsystem and concluded that a runaway
`bthserv` svchost was holding 5 to 11 of the 12 hardware threads, so it filed its own throughput results as a
"floor under that load" and discarded them. **That service did not exist and the results were good.** What the
counters actually report:

| source | pid 1180 (`svchost`, hosting only `bthserv`) |
|---|---|
| `Win32_PerfFormattedData_PerfProc_Process.PercentProcessorTime` | 530 to 1085 "%" |
| `Get-Counter '\Process(*)\% Processor Time'`, single sample | same order (an "NVIDIA app" read 5089 "%") |
| `Win32_PerfRawData_PerfProc_Process.PercentProcessorTime` | 2.2e13 (100-ns units) = **611 CPU-hours**, against 561 hours of uptime |
| `Process.TotalProcessorTime` delta (kernel `GetProcessTimes`) over 4 s | **0.00 cpu-s = 0 %** |

The raw counter is cumulative CPU time since boot, and both the "formatted" class and PDH's single-sample
path present that total as though it were a rate, so a long-lived process reads as hundreds of percent
forever. `\Processor(_Total)\% Idle Time` is pinned at 0 and `% Processor Time` at exactly 100.0 on this
machine for the same reason; even `GetSystemTimes` returns a zero idle delta here, so nothing derived from
system-wide idle accounting can be trusted on this box either. Measured properly, the machine is quiet: the
busiest processes are the editor and the VPN client at 8 to 15 % of one core each, 0.27 to 0.44 of 12 cores
in total.

The rule this leaves: **to measure what else a machine is doing, sample per-process `GetProcessTimes` twice
over a fixed interval and divide by elapsed wall time.** `tools/bench_rounds.ps1` does that and prints
"other load N.NN of 12 cores" next to every step (`docs/data/bench_rounds.log`); it no longer reads the
thermal zone or the clock ratio, which come from the same subsystem and cannot be validated here.

Consequences for what was already filed. The counters were only ever used to *annotate* runs, never inside a
kernel or a timing path, so no measured number in any phase changes; what changes is their interpretation.
The Phase 3.5 throughput results stand as taken on a quiet machine (finding 49). Finding 40's thermal reading
is separately supported by its own evidence (the same kernel measuring 26 GB/s early and 9 to 18 GB/s after an
hour of load, with membw flat), so it is not withdrawn, but any part of it that leaned on a "% Processor
Performance" or "% busy" reading is unsupported. The lesson is the one Phase 3 already wrote down for
throughput and did not apply to the machine itself: pair every number with a measurement you have validated,
and when an annotation says the box is saturated while the box is producing its best-ever bandwidth figure
(31.6 GB/s here, against 28.9 on the "cool" Phase 3 machine), disbelieve the annotation, not the result.

## 48. Q8_K activations meet the promotion criterion; rule 5 is now per activation format

Rule 5's frozen ceilings were what kept Q8_K opt-in in Phase 3 (finding 37): its layer-0 error exceeded a
ceiling frozen from a *different* activation format's noise. Phase 3.5 amends the rule: the ceilings are per
activation format, each frozen at 2 x the per-layer max over the three prompts of that format's own 0..63
measurement, so a format is gated against its own regression and not against another format's noise level
(`tests/fixtures/q8k_ceilings.json` from `docs/data/phase3_5_parity_q8k_0_63.log` via
`tools/freeze_ceilings.py`; `tests/real_layers.rs` and `tests/phase3_e2e.rs` select the file from the active
format, `AQUEDUCT_ACT=q8k|q8_0`, and a format with no frozen file runs ungated as its measurement). The Q8_K
measurement: per-layer relative error 6.7e-4 to 2.2e-3 at layer 0 and up to 5.3e-2 at layer 63 (Q8_0: 8.7e-4
to 2.4e-3 and 1.6e-2), 2 to 4 x Q8_0's, what finding 31 predicted for one scale per 256 values against one per
32; last-position logits vs ref_gguf 0.362 / 0.281 / 0.357 (Q8_0: 0.148 / 0.119 / 0.154), argmax 3/3, top-10
10 / 10 / 9; vs llama.cpp 0.349 / 0.226 / 0.392 (Q8_0 path: 0.443 / 0.317 / 0.364), argmax 3/3, top-10 9 / 10
/ 9. The criterion (argmax 3/3, top-10 >= 9/10, max|diff| below the bf16 noise floor of 0.653 / 0.516 / 0.679,
finding 35) is met on all three prompts, so Q8_K is the production activation for K-quant weights and the
Q8_0-grain path is `aqueduct run --q8-fine` (`matvec::set_q8_fine`). Two facts outside the criterion, filed
for the record: (a) argmax at every prompt position vs ref_gguf is 5/5, 4/4 and 37/39 (Q8_0: 48/48); the e2e
gate now allows a flip only where the reference's token is our runner-up inside the 0.52 noise floor and
prints it (the two `sentence` flips are listed in `docs/data/phase3_timing.txt`); a wrong binding still fails
it. (b) The Q8_K path is now about as far from llama.cpp (0.23 to 0.39) as from the f32 reference (0.28 to
0.36), where the Q8_0 path was nearer to f32 (0.12 to 0.15) than to llama.cpp (0.32 to 0.44): the two engines
now round the same way.

## 49. The kernel gate is met: the production K-quant kernels run at 68 to 89 % of memory bandwidth, and the two-row pass that was supposed to get them there is slower

The Phase 3 report proposed precomputed factor tables plus a two-row pass for the production Q5_K / Q6_K
tail, because those kernels had never exceeded 35 % of membw at the 5120 x 5120 gate shape. Measured on a
machine whose load was measured correctly (finding 47), the premise was wrong: **the kernels were already
memory-bound**. Three rounds of `tools/bench_rounds.ps1` (plugged in, 5 minutes idle, membw then kernels back
to back, every round filed, other load 0.09 to 0.23 of 12 cores throughout) give, at 6 threads and 5120 x
5120, Q5_K **74 to 79 %** of that round's membw and Q6_K **84 to 85 %** (Q4_K 68 to 77 %); at 17408 x 5120,
Q5_K 76 to 85 %, Q6_K 81 to 89 %, Q4_K 75 to 83 % (`docs/data/kernels_bench.txt`). The 50 % gate is met with
room, on the production activation format, at the shape that was failing.

What the two proposed changes actually did:

- **Two rows per pass** (`dot2_*`, both activation forms, checked bit-identical on 900 random row pairs and
  through `matvec` at 1 / 2 / 7 / 64 rows and 1 / 3 / 12 threads) is **slower**: an in-process interleaved
  A/B, the two variants alternated 21 times per line and medians compared, made it slower on 51 of 54 lines,
  ratio 0.65 to 1.16 and mostly 0.85 to 0.95, worst for Q6_K, at one thread as much as at six
  (`docs/data/two_row_ab.log`). That is what a memory-bound kernel should do: sharing the activation loads
  between two rows saves work the machine was not waiting on, while two rows of unpacked codes and two
  accumulator sets exceed the 16 ymm registers. It was removed; rows stay one per pass.
- **The factor tables**, the two-mask Q6_K unpack and the fixed Q5_K high-bit shift are kept and are
  bit-identical to Phase 3. Alternating the Phase 3 and 3.5 binaries at one thread
  (`docs/data/phase3_5_single_thread.log`) puts them inside this box's noise for Q4_K and Q6_K (7.5 to 7.9
  against 7.5 to 7.7, and 5.7 to 6.0 against 5.4 to 5.9 GB/s); the Q8_K Q5_K kernel is the one clear win,
  7.1 to 7.5 against 4.0 to 4.5. The same rewrite applied to the Q8_0-grain Q5_K kernel measured *slower*
  (3.9 to 4.3 against 5.9 to 6.3), so that kernel keeps its Phase 3 body. Neither direction is explained;
  there is no profiler on this box.

One measurement effect in the first (discarded) bench run does not reproduce here: numbers there decayed
within a run, the first configuration reading 18 to 24 GB/s and everything after it 6 to 11. In these three
rounds there is no such decay (round 1 at 5120 runs 21.7, 19.9, 19.5, 21.5, 24.2, 24.8 GB/s in issue order,
ending on its fastest). The likely cause is that the old script's own instrumentation -- a full WMI process
enumeration plus `Get-Counter` calls against a broken counter subsystem, between every step -- was loading
the machine it was trying to observe; that is not proven, and the old numbers are superseded rather than
explained. The single-thread and A/B comparisons above are unaffected either way, because both alternate the
variants under whatever the machine is doing.

## 50. End to end on the production config: 1.15 s per token, 46 % of memory bandwidth, and the remaining 40 % of the token is not matvec

`tests/phase3_e2e.rs` on the production config (Q8_K activations, 6 threads; `docs/data/phase3_timing.txt`,
membw 31.6 GB/s measured in the same session). **Correctness**, unchanged from the parity harness and
identical at 6 and 12 threads: the batched prefill is bit-identical to the sequential feed on every prompt
(hidden, logits, state, cache all 0); last-position logits 0.362 / 0.281 / 0.357 from ref_gguf, exactly the
streamed harness's numbers, against ceilings 0.725 / 0.561 / 0.715; argmax per position 5/5, 4/4, 37/39, the
two `sentence` flips at positions 26 (ours 491 vs ref 685, margin 0.18) and 29 (23540 vs 2627, margin 0.067),
both with the reference token as our runner-up. Greedy vs llama.cpp: `capital` 16/16 (the Phase 3 text
verbatim), `fib` 6/16 where the Q8_0 path had 16/16 (at step 6 ours " 1" beats " 0" by 0.095 and the
continuation is `if n <= 1: return n ... fibonacci(n-1) + fibonacci(n-2)`, a different correct Fibonacci),
`sentence` 5/16 with a 0.024 margin at the split; text coherent on all three. So the Q8_K path loses one
token-equality match the Q8_0 path had by luck, inside finding 42's coin-toss reading.

**Speed**: load 22.4 s; prefill 1.7 to 2.0 tok/s; decode **1.151 / 1.154 / 1.178 s per token = 14.60 / 14.56
/ 14.27 GB/s = 45 to 46 % of membw**; peak RSS 17.975 GB against 17.686 expected, ratio 1.016. The same test
at 12 threads on the same quiet machine gives 1.443 / 1.454 / 1.446 s per token (11.6 GB/s, 37 %), so the
physical-core default is worth 25 % end to end -- more than the 10 % the kernels alone show (finding 51),
because the non-matvec work contends on SMT siblings too. Against Phase 3's best (1.403 to 1.562 s per
token) this is 1.22 to 1.36 x faster per token.

**The >= 60 % end-to-end target is not met, at 45 to 46 %, and the kernels are not the reason.** At 6 threads
they run at 75 to 89 % of membw on the 17408 x 5120 shape (finding 49). At the file's type mix (58 % Q4_K,
33 % Q6_K, 4.5 % Q8_0, 2.3 % Q5_K, 1.3 % Q4_0) and those measured per-type rates, the matvecs of one token
account for about **0.69 s of the measured 1.15 s**. The other ~0.46 s per token, 40 % of the token, is the
work the dot kernels do not cover: 48 DeltaNet recurrence steps of 48 x 128 x 128 f32 (still scalar,
`docs/simd.md`), the causal conv, the norms and the per-token allocations. Closing that is the next speed
item, and it is exactly what the Phase 3 report predicted would become visible once the kernels reached the
bus. A 60 % token needs about 0.89 s, so roughly 0.26 s has to come out of that 0.46 s of non-matvec work.

## 51. Six threads beat twelve on every K-quant kernel and on memory bandwidth itself, so the engine defaults to physical cores

On this 6-core / 12-thread i7-9750H, running the matvec on all 12 hardware threads is slower than running it
on 6 for every K-quant kernel: 35 of the 36 K-quant lines across three rounds, two shapes and both activation
forms (the exception is one tie, 21.97 against 21.99 GB/s). At 5120 x 5120 with Q8_K activations the gap is
Q4_K 68 to 77 % of membw at 6 threads against 64 to 68 % at 12, Q5_K 74 to 79 % against 65 to 69 %, Q6_K 84
to 85 % against 69 to 75 %. The `membw` benchmark itself, which is nothing but AVX2 loads and adds, shows the
same: 29.1 / 32.3 / 31.6 GB/s at 6 threads against 26.8 / 32.3 / 30.6 at 12. Two hardware threads on one core
share its load ports, its L1 and its line-fill buffers; a kernel that is already waiting on DRAM gains no
outstanding misses from the second thread and pays for the contention.

So `aqueduct run` and `bench` default to the **physical** core count, and `--threads N` overrides
(`crates/core/src/cpu.rs`: `GetLogicalProcessorInformationEx(RelationProcessorCore, ..)` on Windows, the
distinct `thread_siblings_list` values under `/sys/devices/system/cpu/cpu*/topology` on Linux,
`available_parallelism` as the fallback everywhere else). The determinism contract is untouched: the pool
partitions rows the same way at any participant count, so nothing about the output moves with the default.
The two `tests/phase3_e2e.rs` runs behind finding 50, at 6 and at 12 threads, agree on every number that is
not a clock: the same logits vs ref_gguf (0.3623 / 0.2807 / 0.3573), the same argmax at every prompt
position, hidden and state max|diff| 0, and the same 32 generated ids for all three prompts.

Two kernels prefer 12 threads, and the decision deliberately ignores them: Q4_0 (26 to 30 % of membw at 6
threads, 29 to 40 % at 12) and Q8_0 weights (52 to 63 % against 56 to 70 %). Both are still compute-bound and
far enough from the bus that a second thread per core finds work to do. Together they are 5.8 % of this file
against 93 % for the K-quants, so the default follows the K-quants; a file with a different mix would want
the opposite, which is what `--threads` is for.

## 52. The decode step now allocates nothing, and taking the allocations out made the token 1.6 x faster

Phase 3.6 item 1: every buffer a decode step touches is preallocated in `model::State` at load
(`layers::Scratch` for the per-layer temporaries, `DeltaScratch` / `AttnScratch` / `MlpScratch` for the
mixers and the MLP, `ActBuf` for the quantised activations, the residual ping-pong, the lm_head input;
`State::reserve(max_pos)` sizes the KV caches and the attention score rows). `tests/decode_alloc.rs` proves
it: a counting global allocator, armed after one warm-up token, records **0 allocations across 32
`forward_token` steps** on the tiny 9-layer model, at 1, 2 and 6 threads. The same file also asserts that the
preallocated path and the old allocating one (`forward_hidden` + `logits_of`) produce bit-identical logits
over those 32 steps, and the real-model gate confirms it at scale: `tests/phase3_e2e.rs` reproduces every
number of the Phase 3.5 run exactly (logits vs ref_gguf 0.3623 / 0.2807 / 0.3573, argmax at every position,
all 96 generated ids), only faster.

The file holds exactly one `#[test]`, deliberately: the allocator counter is global and cargo runs a
binary's tests concurrently, so a second test in the same binary contaminates the count. The first version
of this test had two and reported 301 phantom allocations.

Where the old ones were: about 10,000 per token. Every projection built a fresh `ActVec` whose `Q8Row` /
`Q8KRow` allocated five and four vectors respectively; every decoder layer allocated `normed`, `mixed` and
`h`; every DeltaNet layer allocated seven buffers plus `qn`, `kn` and `o` per head (48 heads x 3); every
attention layer allocated eleven plus `scores` and `probs` per head; the MLP three more.

**The speed.** Decode went from 1.151 / 1.154 / 1.178 s per token to **0.717 / 0.720 / 0.725**, i.e. 23.4 GB/s
= 74 % of the 31.6 GB/s membw, against 45 to 46 % before (`docs/data/phase3_timing.txt`). That is 1.6 x, and
it is more than malloc: to make the head loops allocation-free they had to move off `std::thread::scope`,
which was spawning fresh OS threads for every DeltaNet and every attention layer -- 64 scopes and a few
hundred thread creations **per token** -- onto the persistent pool the matvecs already use. The two changes
were made together and are not separable after the fact, but the profile (finding 53) settles which
dominated: all the work those regions actually do now measures 24 ms per token, so the ~430 ms that
disappeared was overhead, and thread creation is the only candidate of that size.

This also, incidentally, meets the >= 60 % of membw end-to-end target that Phase 3.5 filed as not met.

**Why it matters past speed** (the reason the item was worth doing regardless): Phase 4 runs the engine under
a hard memory cap, and an allocation in the per-token path is where an OOM comes from. A decode step whose
working set is fixed at load cannot fail that way, and its peak RSS is known before the first token.

## 53. Measured, the non-matvec work is 7 % of a token, not 40 %: the engine is a matvec engine now

Phase 3.6 item 2, `docs/data/nonmatvec_profile.txt`, `--features profile`. The profiler charges **exclusive**
wall time per stage (entering a scope charges the elapsed time to the enclosing stage first), so the stages
sum to the token: 756.6 ms of stages against 758 ms measured, 0.2 % apart.

At 6 threads, per token: **matvec 703.7 ms (93.0 %)**, everything else **52.9 ms (7.0 %)** -- deltanet_rec
19.4, conv 13.6, swiglu 11.1, attn 4.9, quantise 2.8, norm 0.46, residual 0.35, rope 0.21, softmax 0.05,
embed 0.004. The matvec figure is 16.84 GB of weights in 703.7 ms = 23.9 GB/s, exactly the rate the kernel
bench measures for these kernels (finding 49), so the dot products are running at their measured capability
and the rest of the token is 53 ms.

Finding 50's 0.46 s of "non-matvec" was an inference from component rates, and it was wrong. The work it
named is 33 ms (deltanet_rec + conv + norm); what was actually there was the per-token allocation and thread
creation that finding 52 removed. **Inferring a breakdown from component rates was the mistake** -- the same
mistake in kind as finding 47's, trusting a derived number over a measured one.

Two things the 1-thread column shows that the 6-thread one cannot. `deltanet_rec` and `attn` scale with
threads (56.1 -> 19.4 ms and 10.5 -> 4.9, about 2.9 x on 6 threads): they are parallel over heads.
`conv` and `swiglu` do not move at all (13.8 -> 13.6, 11.1 -> 11.1): they are serial scalar loops on the
calling thread, and they are 25 of the remaining 53 ms. So the non-matvec work that is left is mostly work
that has never been vectorised or parallelised (`causal_conv1d_step`, `swiglu`, `deltanet_step` are all still
scalar, `docs/simd.md`).

None of it was optimised this session, by instruction, and the number says why that is the right call: the
whole non-matvec budget is 7 % of the token. Driving it to zero would take 0.758 s to 0.705 s. The next real
speed work is not here.
