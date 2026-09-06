# Changelog

## 0.1.0 (2026-09-06)

The first release: a CPU-only engine for Qwen3.8-27B (bartowski's Q4_K_M GGUF) that runs on 8 to 16 GB of
RAM by streaming what does not fit from the drive, with the same greedy output at every memory budget.

What you get:

- `aqueduct doctor`: measures the machine (cores, AVX2, RAM, memory bandwidth, the drive's bus and seek
  penalty, unbuffered read speed), plans the model into the free RAM with or without the model file present,
  predicts the seconds per token plain and with `--spec 3`, and prints a verdict and the exact next commands,
  download and sha256 check included.
- `aqueduct chat`: a multi-turn chat on the model's own template (thinking on or off), streaming tokens,
  sampling from the model's generation defaults or `--greedy`, with the conversation prefix reused across
  turns.
- `aqueduct run`: raw ids or text in, ids and text out, greedy by default; the measurement and test channel.
- `aqueduct plan --budget 8G`: the memory plan for a budget without loading a byte.
- `--budget`: whole layers are pinned in RAM from layer 0 upward while they fit; the rest stream through a
  two-slot ring with prefetch, read unbuffered at the drive's sequential rate, exactly once per token. A budget
  that cannot hold the always-resident set plus the ring is refused with the shortfall in bytes.
- `--spec k`: speculative decoding with the checkpoint's own MTP head, verified in one batched pass, with
  greedy output identical to plain decoding; 2.3 x at an 8 GB laptop's free RAM, 1.8 x at a 16 GB laptop's,
  break-even when the model fits (so plain decoding stays the default).
- AVX2 kernels for Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0 weights with Q8_K activations, bit-identical to their
  scalar references and at every thread count; a blocked GEMM for prefill and verification.
- Release binaries for Windows and Linux compiled for the AVX2 baseline (x86-64-v3); a CPU without AVX2 gets
  one line, not a crash.

What is measured and where: the ladder (`docs/ladder.md`, `docs/data/ladder.txt`), the cost model
(`docs/data/spec_cost_model.txt`), every finding (`docs/payload-vs-doc.md`, 76 items), and the limitations
in `README.md`, which is the section to read first.

Known gaps in this release (details in the README): prompt processing at about 3 tokens per second on AVX2
CPUs; `--spec` pays only while layers stream from disk; a single supported GGUF file; the Linux build compiled
but unrun by its author; no GPU, no vision, no VNNI kernels, no IQ quantisations, no macOS.

The work leading here, phase by phase: reading the checkpoint's files and recording where they disagree with
their documentation (Phase 0); the GGUF reader, a config that refuses to guess, and tokenizer parity (Phase 1);
scalar kernels with hostile fixtures, a tiny same-graph oracle through the real converter, and per-layer parity
on all 64 real layers (Phase 2); the full forward, the AVX2 kernels at memory bandwidth, and an allocation-free
decode step (Phase 3); the memory plan, pinned arenas, the streaming ring and the ladder (Phase 4); the MTP
head, speculative decoding, sampling and the chat template (Phase 5); the blocked verify kernel and the
probe that explains its ceiling (Phase 5.5); the doctor, the README and the release (Phase 6).
