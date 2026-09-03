# Aqueduct — architecture v1

Working name. Rename freely.

## The promise

**Qwen3.8-27B at Q4 quality on a 16 GB laptop, ~10 tok/s. Runs on 8 GB. Same tokens at every budget.**

| machine | quant | target | mode |
|---|---|---|---|
| 8 GB RAM, no GPU | Q4 | runs, ~0.3–0.5 tok/s | floor |
| 16 GB RAM, no GPU | Q4 | ~1–2 tok/s | works |
| 16 GB RAM + 4–6 GB NVIDIA | Q4 | ~10 tok/s | **primary (v2, GPU tier)** |
| 16 GB RAM + 4–6 GB NVIDIA | IQ2 | ~25 tok/s | speed mode (v2, labelled with quality cost) |
| 32 GB RAM | Q4 | fits fully | reference |

v1 ships the CPU + disk engine (rows 1–2). v2 adds the GPU tier (rows 3–4). Do not
start v2 before v1 is on crates.io.

## What is borrowed from kimi-k3-in-c, and what is not

Borrowed:
- Layer-contiguous packed weights, 4 KB aligned, read with O_DIRECT
- Pinned prefix + one ring slot + prefetch of layer L+1 (a cyclic scan defeats LRU; pin instead)
- 2 MB hugepages / large pages for the arenas
- Memory plan summed before any allocation; peak RSS measured after; refuse to start if it will not fit
- Config reader that refuses to substitute defaults
- Fixtures designed to fail a specific plausible wrong implementation
- Tiny same-graph oracle, then per-layer parity on real weights, then full logit parity
- `--ids` channel: ids in, ids out, no tokenizer, for reproducible tests
- The delta-rule recurrence structure (Kimi KDA and Qwen Gated DeltaNet are the same family)

Not borrowed:
- Expert cache, expert streaming, MXFP4 nibble matmul (dense model, nothing to skip)
- Double-precision bit-identical kernels (half throughput; f32 accumulators here)
- Greedy-only (we ship sampling + chat template; greedy is the test mode)

## Non-goals for v1

- GPU. Vision tower. Batched multi-user serving. Fine-tuning. macOS.
- Beating llama.cpp when the model already fits in RAM. Our edge is only when it does not.

## Model facts — VERIFY in Phase 0, never hardcode

Known from public docs, to be confirmed against `config.json` and the tensor index:
- 27B dense, 64 layers, `full_attention_interval: 4` → 16 full-attention (GQA) layers, 48 Gated DeltaNet layers
- Architecture `Qwen3_5ForConditionalGeneration`; text config nested under `text_config`; `vision_config` present (skip it)
- Built-in MTP draft head in the checkpoint
- 262K native context. Chat template opens assistant turns with `<think>`; thinking can be disabled per request
- Q4_K_M GGUF ≈ 17 GB (unsloth build). bf16 ≈ 56 GB

Unknown until Phase 0: hidden size, head counts, head dims, DeltaNet conv width, vocab, tie of embed/lm_head,
exact MTP head tensor layout, whether GGUF layer tensors are contiguous on disk, RMSNorm eps, RoPE params for the
GQA layers, whether GGUF already strips the vision tower.

## Language and platform

- Rust. Windows + Linux from day one (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT` behind one trait).
- No BLAS, no ggml, no candle for the hot path. `tokenizers` crate is acceptable for BPE (verify parity anyway).
- Single binary + one library crate. Headless core, CLI shell, later a tiny HTTP server.

## Components

```
aqueduct/
  crates/core/
    gguf.rs        GGUF reader: header, tensor index (name, type, shape, offset), no data copy
    config.rs      refuse-to-guess config; one struct is the model contract
    tok.rs         tokenizer + parity test harness
    pack.rs        optional repack into layer-contiguous trunk (only if Phase 0 says GGUF is not contiguous)
    tier.rs        memory plan, pinned prefix, ring slot, prefetch, O_DIRECT reads
    kernels/       q4k_q8 dot, rmsnorm, rope, softmax, silu/swiglu, deltanet_step
    layers/        gated_deltanet.rs, gqa_attention.rs, mlp.rs, mtp.rs
    forward.rs     embed → 64 layers → norm → lm_head; incremental state; KV cache for the 16 GQA layers
    spec.rs        MTP drafting + batched greedy/sampled verification
    sample.rs      greedy, temperature, top-p, top-k
  crates/cli/      aqueduct run / doctor / bench / pack
  tools/           python: ref_forward.py, emit_fixtures.py, cmp_logits.py, sim_ladder.py
  tests/fixtures/  captured payloads, tiny oracle checkpoint, per-kernel fixtures
  scripts/         doctor (measures disk the way the engine reads it), ladder runner (cgroup / job object)
  docs/data/       every number in the README comes from a file here
```

## Memory tiers (v1: two tiers, designed for three)

```
tier 0  (v2) VRAM    ~200 GB/s   first N0 layers
tier 1       RAM      ~50 GB/s   next N1 layers, pinned once at start
tier 2       NVMe    ~3–6 GB/s   remaining layers, streamed through a ring slot each token
```

Policy: fill the fastest tier first, in layer order. Layer L lives in exactly one tier for the run.
The ring slot is sized to the largest streamed layer. Prefetch L+1 while computing L.
Everything not per-layer (embeddings, lm_head, norms, MTP head) is pinned in RAM always.

Per-token cost model, printed by `doctor` before download:
`ms/token ≈ 4 × GB_vram + 20 × GB_ram + 200 × GB_nvme` (laptop-class numbers; doctor measures the real ones).

## Kernels

- Weight format: Q4_K (super-blocks of 256, 6-bit scales/mins) as shipped in Q4_K_M GGUF. Activations quantised
  to Q8 per row before the dot product. Int8 multiply via `_mm256_maddubs_epi16` on AVX2; scalar fallback;
  NEON later. This is the llama.cpp approach and it is the correct one for CPU.
- f32 accumulators. Bit-identity across thread counts is a nice-to-have, not a contract.
- Gated DeltaNet step: decay → read → rank-one delta write → read. Parallel over heads. Fixed-size state per layer.
- GQA attention on 16 layers with a growing KV cache; RoPE on those layers only (verify in Phase 0).
- Short causal conv before DeltaNet projections (verify width).
- MTP head: forward it after the main stack to draft k tokens; verify in one batched forward.

## Determinism contract

Greedy decode must emit identical token ids at every memory budget and every thread count.
That is the property the whole ladder table depends on. Sampling is seeded and reproducible per seed.

## Phases and gates

One phase per working session. Gate is verified by hand before commit. No forward-bleeding.

### Phase 0 — payloads before structs
- Download: `config.json`, `tokenizer.json`, `generation_config.json`, chat template; the Q4_K_M GGUF;
  the bf16 safetensors index (`model.safetensors.index.json`) only, not the shards.
- Capture: full GGUF tensor index (name/type/shape/offset/bytes) to `tests/fixtures/gguf_index.json`.
- Check and record: are layer L's tensors contiguous in the GGUF? Is the vision tower present? Is embed tied?
  MTP head tensor names and shapes. Every number in "Model facts".
- Reference: `tools/ref_forward.py` runs HF transformers on 3 fixed prompts, dumps logits at the last position
  and per-layer hidden-state max-abs. Keep the prompts tiny; run on the 32 GB box, CPU is fine.
- Write `docs/payload-vs-doc.md`: where reality diverged from the model card.
- **Gate:** fixtures committed; payload-vs-doc written; you have read every tensor name once.

### Phase 1 — read the model
- GGUF reader, config reader (refuses defaults), tokenizer with parity harness (45 cases incl. CJK, emoji, code).
- **Gate:** tokenizer parity 45/45 vs HF; config struct printed matches Phase 0 fixtures; index of all tensors
  built in < 1 s with no data read.

### Phase 2 — kernels and one layer
- Q4_K×Q8 dot, rmsnorm, rope, swiglu, DeltaNet step, GQA attention. Per-kernel fixtures with hostile inputs.
- Tiny oracle: a 9-layer model with the same graph (two full blocks of the 3:1 pattern plus one more),
  hidden 64, generated by `tools/make_tiny_checkpoint.py`, with a PyTorch reference.
- Run layer 0..N of the real model against `ref_forward.py` per-layer dumps.
- **Gate:** every kernel fixture passes; tiny oracle greedy 20/20 tokens match; real layers 0–7 within
  error budget (budget derived from width, as in Kimi: a wrong binding misses by ~1, not 1e-6).

### Phase 3 — full forward, fully resident
- All 64 layers, embed, lm_head. Greedy. Runs on the 32 GB box with everything in RAM.
- Incremental decode (KV + carried DeltaNet state) and full-recompute paths, both gated.
- **Gate:** logit parity vs reference on 3 prompts (argmax match, top-10 overlap, max|diff| under budget);
  incremental == recompute token-for-token for 32 generated tokens; output is coherent text.

### Phase 4 — tiers
- Memory plan, pinned prefix, ring slot, prefetch, O_DIRECT on both OSes, large pages.
- `pack` subcommand only if Phase 0 showed the GGUF is not layer-contiguous.
- Ladder runner: Linux `systemd-run -p MemoryMax=`, Windows job object with memory limit.
- **Gate:** identical token ids at 6 / 8 / 12 / 16 / 32 GB caps; ladder table in `docs/data/`; peak RSS lands
  on budget at every rung; measured s/token within 2× of the cost model.

### Phase 5 — speed and usability
- MTP speculative decode with batched verification. Sampling. Chat template with thinking on/off.
- **Gate:** greedy output with `--spec` identical to without; acceptance rate and speedup measured and filed;
  a real chat turn works end to end from the CLI.

### Phase 6 — ship
- `doctor` measures disk (random reads at the engine's size, queue depth 1 and 8, not `dd`), sizes RAM to a
  preset, prints expected s/token and the exact next command.
- README with the ladder, the cost model, and an honest-misses section. crates.io + GitHub release binaries
  for Windows and Linux. Demo GIF.
- **Gate:** clean-clone → doctor → download → run works on a machine that is not yours.

### v2 (after ship, separate planning doc)
- Tier 0: CUDA offload of the first N0 layers. This is what turns the 16 GB laptop into ~10 tok/s.
- DFlash2 drafter (7 draft tokens) if acceptance holds; IQ2 speed mode with published quality cost.
- Activation sparsity is a research question, not a roadmap item.

## Risks to write down now

- **Disk, not RAM, is the gate for end users.** Many 8–16 GB laptops have SATA SSDs. `doctor` must say so
  before they download 17 GB. Publish numbers for SATA and NVMe separately.
- **Free RAM on a 16 GB Windows laptop is ~11 GB, on 8 GB it is ~5 GB.** Plan for those; advertise the nominal size.
- **Gated DeltaNet ordering** (norm/gate/proj, which of q/k get L2-normed, where the query scale goes) is where
  fluent-but-wrong models come from. Every one of those is a fixture, not an assumption.
- **The MTP head layout** is undocumented for CPU engines. Phase 0 captures it; Phase 5 may need a day of reverse-engineering.
- **We are not user zero on the target machine.** Cgroup/job-object caps stand in for it until a real 16 GB test box exists. Borrow one before Phase 6.

## The one-sentence pitch

"Run Qwen3.8-27B at full Q4 quality on a 16 GB laptop — it streams what doesn't fit instead of forcing you to a worse quant."
