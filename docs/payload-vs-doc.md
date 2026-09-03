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
