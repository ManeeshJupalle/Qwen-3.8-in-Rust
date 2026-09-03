# Config mapping: GGUF key -> HF config.json path -> `ModelConfig` field

The engine reads only the GGUF. `config.json` (`tests/fixtures/config.json`, verbatim from `Qwen/Qwen3.8-27B`)
is a test fixture; `crates/core/tests/config_vs_fixture.rs` checks every row below against it by JSON path.
The same table lives in code as `aqueduct_core::config::MAPPING`, and a test asserts each of its keys, fields
and paths appears in this file.

`{arch}` is the value of `general.architecture`, which is `qwen35` for this model (HF calls it `qwen3_5`).
A missing or mistyped key is a hard error naming the key (`GgufError::MissingKey` / `WrongType`); no field has a
default. Derived fields are computed once in `ModelConfig::from_gguf` from other keys and the tensor index
(names and shapes only; no tensor data is read).

## Fields read from GGUF metadata

| GGUF key | HF config.json path | `ModelConfig` field | note |
|---|---|---|---|
| `general.architecture` | `model_type` | `architecture` | qwen35 vs qwen3_5: same architecture, different spelling; compared after removing '_' |
| `general.name` | - | `name` | display only |
| `general.file_type` | - | `file_type` | 15 = Q4_K_M recipe label, not the tensor types |
| `general.quantization_version` | - | `quantization_version` | |
| `{arch}.block_count` | `text_config.num_hidden_layers + text_config.mtp_num_hidden_layers` | `block_count` | 65 = 64 + 1 MTP block |
| `{arch}.nextn_predict_layers` | `text_config.mtp_num_hidden_layers` | `nextn_predict_layers` | |
| `{arch}.context_length` | `text_config.max_position_embeddings` | `context_length` | |
| `{arch}.embedding_length` | `text_config.hidden_size` | `hidden_size` | |
| `{arch}.feed_forward_length` | `text_config.intermediate_size` | `intermediate_size` | |
| `{arch}.attention.head_count` | `text_config.num_attention_heads` | `n_head` | |
| `{arch}.attention.head_count_kv` | `text_config.num_key_value_heads` | `n_head_kv` | |
| `{arch}.attention.key_length` | `text_config.head_dim` | `head_dim_k` | |
| `{arch}.attention.value_length` | `text_config.head_dim` | `head_dim_v` | |
| `{arch}.attention.layer_norm_rms_epsilon` | `text_config.rms_norm_eps` | `rms_norm_eps` | f32 in GGUF, f64 in JSON; compared as f32 |
| `{arch}.full_attention_interval` | `text_config.full_attention_interval` | `full_attention_interval` | |
| `{arch}.rope.freq_base` | `text_config.rope_parameters.rope_theta` | `rope_freq_base` | |
| `{arch}.rope.dimension_count` | `text_config.head_dim * text_config.partial_rotary_factor` | `rope_dim` | 64 = 256 * 0.25 |
| `{arch}.rope.dimension_sections` | `text_config.rope_parameters.mrope_section` | `rope_sections` | GGUF has a 4th entry (0); first 3 compared |
| `{arch}.ssm.conv_kernel` | `text_config.linear_conv_kernel_dim` | `dn_conv_kernel` | |
| `{arch}.ssm.state_size` | `text_config.linear_key_head_dim` | `dn_head_dim_k` | Mamba name for the DeltaNet head dim |
| `{arch}.ssm.group_count` | `text_config.linear_num_key_heads` | `dn_n_k_heads` | |
| `{arch}.ssm.time_step_rank` | `text_config.linear_num_value_heads` | `dn_n_v_heads` | |
| `{arch}.ssm.inner_size` | `text_config.linear_num_value_heads * text_config.linear_value_head_dim` | `dn_inner_size` | 6144 = 48 * 128 |
| `tokenizer.ggml.bos_token_id` | `text_config.bos_token_id` | `bos_id` | HF `tokenizer_config.json` has no BOS token; GGUF names 248044 anyway |
| `tokenizer.ggml.eos_token_id` | `generation_config.json eos_token_id[0]; text_config.eos_token_id differs (finding)` | `eos_ids` | GGUF lists one EOS; generation_config lists two |
| `tokenizer.ggml.padding_token_id` | `tokenizer_config.json pad_token (id)` | `pad_id` | text_config.pad_token_id is null |
| `tokenizer.ggml.add_bos_token` | `tokenizer_config.json add_bos_token` | `add_bos` | |
| `tokenizer.ggml.model` | `tokenizer.json model.type (BPE) / tokenizer_config tokenizer_class` | `tokenizer_model` | gpt2 = byte-level BPE |
| `tokenizer.ggml.pre` | - | `tokenizer_pre` | pre-tokenizer regex id used by llama.cpp |

## Derived fields (from the keys above and the tensor index)

| derivation | HF config.json path | `ModelConfig` field | note |
|---|---|---|---|
| `derived: block_count - nextn_predict_layers` | `text_config.num_hidden_layers` | `n_layer` | |
| `derived: len(tokenizer.ggml.tokens), checked against token_embd.weight shape` | `text_config.vocab_size` | `vocab_size` | padded vocab |
| `derived: ssm.inner_size / ssm.time_step_rank` | `text_config.linear_value_head_dim` | `dn_head_dim_v` | |
| `derived: blk.N.attn_q.weight present <-> (N+1) % full_attention_interval == 0` | `text_config.layer_types` | `full_attention_layers / deltanet_layers` | 0-based; error if tensors and interval disagree |
| `derived: attention.head_count / attention.head_count_kv` | `text_config.num_attention_heads / text_config.num_key_value_heads` | `gqa_group` | |
| `derived: blk.N.attn_q.weight rows == 2 * head_count * key_length` | `text_config.attn_output_gate` | `attn_output_gate` | from the tensor shape, not a KV key |
| `derived: output.weight present` | `text_config.tie_word_embeddings (negated)` | `tie_word_embeddings` | |
| `derived: no dedicated MTP embedding/output tensors` | `text_config.mtp_use_dedicated_embeddings` | `mtp_dedicated_embeddings` | |
| `derived: n_v_heads * head_dim_k * head_dim_v` | - | `dn_state_elems_per_layer` | DeltaNet recurrent state, f32 elements |
| `derived: (2*k_dim + v_dim) * (conv_kernel - 1)` | - | `dn_conv_state_elems_per_layer` | causal conv carry-over |
| `derived: max layer span over layers 0..n_layer` | - | `ring_slot_bytes` | largest streamed layer |
| `derived: non-layer tensors + MTP blocks` | - | `pinned_bytes` | embed, output, output_norm, blk.64 |

Other derived values: `mtp_layers` (block indices `n_layer..block_count`), `q_dim`, `kv_dim`, `dn_k_dim`,
`dn_v_dim`, `dn_qkv_dim` (checked against the `attn_q`, `attn_k`, `attn_qkv`, `ssm_conv1d`, `ssm_a` shapes of
every layer), `layer_bytes` (span of every block), `largest_streamed_layer`.

## Consistency checks performed in `from_gguf` (each a hard error)

- `nextn_predict_layers < block_count`; `head_count % head_count_kv == 0`; `ssm.inner_size % ssm.time_step_rank == 0`;
  `ssm.time_step_rank % ssm.group_count == 0`; `rope.dimension_count <= key_length`; `2 * sum(rope.dimension_sections) == rope.dimension_count`.
- `token_embd.weight` shape is `[embedding_length, len(tokens)]`; `output.weight` (if present) has the same shape; `output_norm.weight` exists.
- Every layer `0..n_layer` has exactly one of `attn_q.weight` / `ssm_a`, and which one agrees with the interval rule.
- Full-attention layers: `attn_q` is `[hidden, q_dim]` or `[hidden, 2*q_dim]` (consistently across layers), `attn_k` is `[hidden, kv_dim]`.
- DeltaNet layers: `attn_qkv` is `[hidden, 2*k_dim + v_dim]`, `ssm_conv1d` is `[conv_kernel, 2*k_dim + v_dim]`, `ssm_a` is `[n_v_heads]`.
- Every MTP block has `nextn.eh_proj.weight`.
- `general.alignment` is absent from this file; the GGUF specification defines 32 in that case (this is a file-format
  rule, not a model default). `read_into` additionally asserts every tensor offset is a multiple of 32.

## Values config.json has that the GGUF does not (findings; see docs/payload-vs-doc.md items 25 and 26)

These are NOT fields of `ModelConfig`. They are needed by Phase 2 and have no GGUF key, so they were not
hardcoded; how to carry them is an open question for the user.

| HF path | value | why the engine needs it |
|---|---|---|
| `text_config.hidden_act` | `silu` | MLP activation (llama.cpp hardcodes SiLU per architecture) |
| `text_config.output_gate_type` | `swish` | DeltaNet output gate activation |
| `text_config.mamba_ssm_dtype` | `float32` | precision of the DeltaNet recurrent state |
| `text_config.rope_parameters.mrope_interleaved` | `true` | only matters with image/video positions; text-only collapses to plain RoPE |
| `text_config.rope_parameters.rope_type` | `default` | no scaling; llama.cpp defaults to freq_scale 1 when no key exists |
| `generation_config.json eos_token_id[1]` | 248044 `<\|endoftext\|>` | second stop id; GGUF only names `<\|im_end\|>` as EOS (248044 is its bos/pad id) |
