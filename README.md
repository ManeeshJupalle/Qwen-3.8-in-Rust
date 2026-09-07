# Qwen-3.8-in-Rust

`aqueduct` runs Qwen3.8-27B (bartowski's Q4_K_M GGUF, 17.8 GB) on a CPU-only Windows or Linux machine with
8 to 16 GB of RAM, streaming the layers that do not fit in RAM from the drive on every token, and it emits
the same greedy tokens at every memory budget. It is written in Rust; the tier design is borrowed from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c).

> Run Qwen3.8-27B at full Q4 quality on a 16 GB laptop: it streams what doesn't fit instead of forcing you
> to a worse quant.

It is slow. On the laptop it was built on, a 16 GB machine's free RAM gives about 0.6 tokens per second and
an 8 GB machine's about 0.5, both with `--spec 3`. The table below is the whole story; the
[limitations](#limitations-and-misses) section is the part to read before downloading 17.8 GB.

## The ladder

Three prompts, 32 greedy tokens each, at memory budgets enforced by a Windows job object on one machine.
The first two rows are the free RAM of an 8 GB and a 16 GB Windows laptop. `--spec 3` drafts three tokens
per round with the model's own MTP head and verifies them in one batched pass; the verified output is
identical to plain decoding, so the speed-up is free of quality cost. Every number is in the linked file.

| budget | plain s/token | `--spec 3` s/token | speed-up | peak RSS plain / spec (GB) | disk GB/token | source |
|---|---|---|---|---|---|---|
| **5 GiB** (an 8 GB laptop's free RAM) | 4.305 | 1.894 | 2.27 x | 5.103 / 5.057 | 13.406 | [ladder.txt](docs/data/ladder.txt) |
| **11 GiB** (a 16 GB laptop's free RAM) | 2.819 | 1.581 | 1.78 x | 11.569 / 11.557 | 7.008 | [ladder.txt](docs/data/ladder.txt) |
| 6 GiB | 4.095 | 1.979 | 2.07 x | 6.051 / 6.005 | 12.461 | [ladder_phase5.txt](docs/data/ladder_phase5.txt) ¹ |
| 8 GiB | 3.628 | 1.812 | 2.00 x | 8.218 / 8.172 | 10.298 | [ladder_phase5.txt](docs/data/ladder_phase5.txt) ¹ ² |
| 12 GiB | 2.690 | 2.050 | 1.31 x | 12.535 / 12.504 | 5.991 | [ladder_phase5.txt](docs/data/ladder_phase5.txt) ¹ |
| 16 GiB | 1.572 | 1.344 | 1.17 x | 16.946 / 16.860 | 1.588 | [ladder.txt](docs/data/ladder.txt) ³ |
| resident (everything in RAM) | 1.170 | 1.178 | 0.99 x | 17.961 / 18.127 | 0 | [ladder.txt](docs/data/ladder.txt) ³ |

¹ Measured with the Phase 5 verify kernel; the current (blocked) kernel makes the `--spec` column faster on
streamed budgets (compare 5 GiB: 2.13 x then, 2.27 x now; [ladder_phase5.txt](docs/data/ladder_phase5.txt)
against [ladder.txt](docs/data/ladder.txt)). ² Re-measured with the current kernel over 200 tokens: 3.658 s/token
plain, 1.644 with `--spec 3`, 2.23 x ([spec_identity_8g.txt](docs/data/spec_identity_8g.txt)). ³ Taken with the
laptop hot; cool, the same rungs measured 1.155 / 1.129 s/token at 16 GiB and 0.830 / 1.057 resident
([ladder_phase5.txt](docs/data/ladder_phase5.txt)), and the plain resident token 0.717 s at best
([phase3_timing.txt](docs/data/phase3_timing.txt)).

**The machine.** One laptop: Intel i7-9750H (6 cores / 12 threads, AVX2, no AVX-512), 32 GB DDR4-2667, the
GGUF on a Crucial P310 NVMe SSD, Windows 11. Memory bandwidth as the kernels see it: 24 to 31 GB/s depending
on how hot the machine is ([membw.txt](docs/data/membw.txt), [doctor_maneesh-msi.txt](docs/data/doctor_maneesh-msi.txt),
[kernels_bench.txt](docs/data/kernels_bench.txt)); unbuffered sequential reads 2.95 GB/s at queue depth 1 and
3.30 at depth 2 ([doctor_maneesh-msi.txt](docs/data/doctor_maneesh-msi.txt)). The 8 GB and 16 GB rows are memory
caps on that 32 GB box, not measurements on real 8 GB or 16 GB laptops.

**The thermal caveat.** This laptop throttles compute by up to 2.5 x within an hour of load while its memory
bandwidth barely moves (finding 40 in [payload-vs-doc.md](docs/payload-vs-doc.md), [kernels_bench.txt](docs/data/kernels_bench.txt)).
Disk-bound rungs (5 to 12 GiB) do not care; RAM-bound rungs (16 GiB, resident) swing by 1.4 x between a cool
and a hot run, which is why two numbers are given for them. Every speed-up in the table pairs a plain and a
`--spec` run of the same prompt taken minutes apart, so the ratios hold in either state.

**Identical output tokens at every row.** The 96 ids (3 prompts x 32 tokens) are the same at every budget,
with and without `--spec`, and equal to the fully resident run three phases earlier:
[ladder_expected_ids.json](tests/fixtures/ladder_expected_ids.json), checked in [ladder.txt](docs/data/ladder.txt).

## Versus llama.cpp

Same file, same laptop, the same three prompts and 32 greedy tokens (ids in, ids out), both engines. llama.cpp is
the official release build b10827 (`llama-b10827-bin-win-cpu-x64.zip`, the AVX2 backend it picks on this CPU),
`-t 6`, run through `llama-server`; the memory caps are the ladder's, applied from outside by
[cap_run.ps1](scripts/cap_run.ps1): a Windows job object with commit limits and a hard working-set maximum,
both equal to the cap (a commit cap alone is blind to a memory-mapped model). Every number, the ids and the
failure lines are in [vs_llamacpp.txt](docs/data/vs_llamacpp.txt) ([vs_llamacpp.ps1](scripts/vs_llamacpp.ps1)
is the run). The aqueduct columns give the filed ladder and a rerun taken in the same session as the llama.cpp
sweep with the laptop hot ([ladder_phase61.txt](docs/data/ladder_phase61.txt)); the machine moved 11 GB/s
through RAM in that rerun against 20 in the filed ladder, and the numbers follow. A short `llama-bench` (32
tokens, 2 repetitions) taken straight after that rerun measured 0.16 tokens/s, 6.3 s/token, against the rerun's
resident 1.53: the ratio holds in the same thermal state.

| memory | aqueduct plain, filed / hot rerun | aqueduct `--spec 3`, filed / hot rerun | llama.cpp as shipped | llama.cpp `--no-repack` |
|---|---|---|---|---|
| resident | 1.170 / 1.534 s/token (0.830 cool) | 1.178 / 1.446 | 3.88 s/token; `llama-bench` tg128 5.26 (0.19 tok/s) | 3.86 |
| 11 GiB (a 16 GB laptop's free RAM) | 2.819 / 3.379 | 1.581 / 2.384 | 9.22 on the two short prompts, then dies on the 39-token prompt (`bad allocation`) | 12.09 ¹ |
| 8 GiB | 3.628 / 4.020 | 1.812 / 2.491 | does not load (`unable to allocate CPU_REPACK buffer`, 9.6 GB) | 12.39 ¹ |
| 5 GiB (an 8 GB laptop's free RAM) | 4.305 / 4.551 | 1.894 / 2.336 | does not load (the same 9.6 GB buffer) | 13.42 ¹ |

¹ A floor, not the number a real 8 or 16 GB laptop would see. The working-set cap trims llama.cpp's mapped pages,
but on this 32 GB box they stay in the OS standby list and come back by soft fault: the drive delivered 0.004,
0.004 and 0.025 GB per token in those three runs, where a machine with only that much RAM has to read 7, 10 and
13 GB per token (the ladder's disk column), 2 to 4 s per token more at this drive's 3.3 GB/s.

How it comes out: this engine is faster at every row, by 2.5 x resident (3.4 x against `llama-bench`) and by 2.9 x
to 3.6 x under the caps (plain, hot rerun against the no-repack floor; 3.1 x to 4.3 x against the filed ladder;
`--spec 3` widens it). As shipped, llama.cpp cannot run this model in 8 or 5 GiB at all: its CPU backend copies
the Q4_K weights into a 9.6 GB repacked buffer (11.9 GB committed in all) on top of the 16.5 GiB mapped file,
24.7 GB of working set for "resident", so it pages even on the 32 GB laptop; at 11 GiB it loads, decodes at
9.2 s/token, and dies on the longer prompt when its batch buffers push the commit past the cap. `--no-repack`
keeps the weights in the mapped file (2.2 GB committed) and runs under every cap at 12 to 13 s/token. Where
llama.cpp is ahead: prompt processing, 3.77 tokens/s at 512 tokens (`llama-bench` pp512) against this engine's
roughly 3 (limitation 1); and with `--no-repack` it needs no memory plan, running under any working-set cap at
whatever speed the paging allows, where this engine refuses a budget its plan does not fit. The greedy ids:
llama.cpp as shipped matches this engine on 32/32, 6/32 and 32/32 tokens of the three prompts, and with
`--no-repack` on 32/32, 6/32 and 5/32, so its own two kernel paths part at token 5 of the prose prompt
(limitation 5). The figure this section replaces, 4.2 to 4.7 s/token through llama-cpp-python, is retired
(finding 21).

## Quickstart

You need an x86-64 CPU with AVX2 (Intel Haswell 2013 / AMD Excavator 2015 or newer), 5 GiB of free RAM or more,
18 GB of free disk on an NVMe drive (a SATA SSD works and is slower; a spinning disk is not recommended), and
about 12 MB of downloads besides the 17.8 GB model. `aqueduct doctor` checks all of that before you download.

### Windows

1. Download `aqueduct-v0.1.0-x86_64-pc-windows-msvc.exe` from the
   [release page](https://github.com/ManeeshJupalle/Qwen-3.8-in-Rust/releases/tag/v0.1.0), check its sha256 against
   the one on that page (`certutil -hashfile <file> SHA256`; also filed in
   [release_v0.1.0_sha256.txt](docs/data/release_v0.1.0_sha256.txt)), rename it to `aqueduct.exe`, put it in an
   empty folder and open a terminal there.
2. Run the doctor. It measures the machine, plans the model into your free RAM, predicts the speed, gives a
   verdict and prints the exact commands for the next steps with the paths filled in:
   ```
   aqueduct doctor
   ```
3. Download the model (17.8 GB) and verify it. Only this file is supported; the expected sha256 is
   `e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b`:
   ```
   curl.exe -L --create-dirs -o C:\models\Qwen3.8-27B-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf
   certutil -hashfile C:\models\Qwen3.8-27B-Q4_K_M.gguf SHA256
   ```
4. Download the tokenizer, the chat template and the generation defaults (Qwen's files, Apache-2.0) into
   `models\Qwen3.8-27B\` under the folder you run from:
   ```
   curl.exe -L --create-dirs -o models\Qwen3.8-27B\tokenizer.json https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json
   curl.exe -L --create-dirs -o models\Qwen3.8-27B\chat_template.jinja https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/chat_template.jinja
   curl.exe -L --create-dirs -o models\Qwen3.8-27B\generation_config.json https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/generation_config.json
   ```
5. Chat, with the budget the doctor recommended (its `--budget` is your free RAM rounded down; `--spec 3`
   where the doctor says it pays, which is whenever layers stream from disk):
   ```
   aqueduct chat --model C:\models\Qwen3.8-27B-Q4_K_M.gguf --budget 11G --spec 3
   ```
   `/quit` ends the session, `--no-think` skips the model's thinking block, `--greedy` makes the output
   deterministic, `aqueduct chat --show-config --budget 11G` prints the memory plan without loading anything.

### Linux

The same steps with `aqueduct-v0.1.0-x86_64-unknown-linux-gnu` (`chmod +x`, `sha256sum -c`), the model at
`models/Qwen3.8-27B-Q4_K_M.gguf` (the default path there) and forward slashes:
```
./aqueduct doctor
curl -L --create-dirs -o models/Qwen3.8-27B-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf
echo "e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b  models/Qwen3.8-27B-Q4_K_M.gguf" | sha256sum -c -
for f in tokenizer.json chat_template.jinja generation_config.json; do curl -L --create-dirs -o models/Qwen3.8-27B/$f https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/$f; done
./aqueduct chat --budget 11G --spec 3
```
The Linux build has never been run by its author (see the limitations); the binary is built on GitHub's
`ubuntu-latest` and needs that image's glibc or newer.

### Build from source

Rust 1.87 or newer; no other toolchain, no BLAS, no Python.
```
git clone https://github.com/ManeeshJupalle/Qwen-3.8-in-Rust.git
cd Qwen-3.8-in-Rust
cargo build --release
target\release\aqueduct doctor        (Windows; cmd.exe does not run a path written with forward slashes)
target/release/aqueduct doctor        (Linux)
```
A release build from a fresh clone took 91 s on the reference laptop.
`cargo build --release` compiles for baseline x86-64 and selects the AVX2 kernels at run time; that is the
build every number above was measured with. The release binaries add `-C target-cpu=x86-64-v3` (everything
compiled for AVX2, not only the kernels); the path from the process entry to the `cpuid` check was audited
instruction by instruction to contain no AVX or BMI code ([avx2_gate_audit.txt](docs/data/avx2_gate_audit.txt),
[tools/asm_audit.py](tools/asm_audit.py)), so a CPU without AVX2 gets one line and exit code 2, but that build
was not re-measured. `cargo test` runs 80 tests (11 of them skip, and say so, when the 17.8 GB file is absent;
the tiny oracle model and the tokenizer files the rest need are what CI downloads, see
[ci.yml](.github/workflows/ci.yml)); the two model-scale gates are `#[ignore]`.

## How it works

The memory plan comes first: from the GGUF header alone (no tensor bytes), the engine sums what must always be
resident (the embedding table, the output head, the norms, the MTP block, the DeltaNet state, the KV cache for
`--max-pos` positions, a 256 MiB baseline reserve) and a ring of two slots sized for the largest streamed
layer, then pins whole layers from layer 0 upward while they fit the budget. It prints the table, refuses a
budget it cannot meet with the shortfall in bytes, and only then allocates ([docs/tiers.md](docs/tiers.md)).

Pinned layers are read once at load with unbuffered sequential reads (`FILE_FLAG_NO_BUFFERING` on Windows,
`O_DIRECT` on Linux; 7 s for the whole file on the NVMe). The rest are streamed: one I/O thread walks the
streamed layers in order into the ring, prefetching layer L+1 while layer L computes, so a token costs one
sequential pass over the streamed bytes at the drive's rate, which is what the disk-bound rows of the ladder
measure (the drive is busy 94 to 97 % of the wall time there). Every weight byte from disk is read exactly
once per token and the engine asserts it.

Decoding is one weight pass per token: Q8_K-quantised activations against the file's Q4_K / Q5_K / Q6_K rows
with AVX2 integer kernels in ggml's structure, f32 accumulators, no fast-math, the same result at every thread
count ([docs/kquant-dot.md](docs/kquant-dot.md), [docs/simd.md](docs/simd.md)). The 48 Gated DeltaNet layers
carry a fixed-size state and the 16 GQA layers a KV cache ([docs/deltanet.md](docs/deltanet.md),
[docs/rope.md](docs/rope.md)).

`--spec k` uses the checkpoint's own MTP head to draft k tokens, then verifies them with one batched pass of
k+1 rows through all layers, so a round costs one disk pass for up to k+1 tokens; the accepted tokens are
exactly what plain greedy decoding would have emitted, and a rollback restores the DeltaNet states from one
snapshot plus a replay of saved inputs, without a second weight pass ([docs/spec.md](docs/spec.md),
[docs/mtp.md](docs/mtp.md)). Greedy output is identical at every budget, every thread count and every k;
sampling is seeded and reproducible per seed.

## The cost model

`aqueduct doctor` predicts the token time from two measurements of your machine and three constants from this one:

```
t_plain = bytes_ram / membw + bytes_disk / diskbw + 0.053 s
t_round = t_plain + 0.085 s x k + 0.527 s x (k + 1);   t_spec = t_round / 2.74 tokens per round
```

`bytes_ram` and `bytes_disk` come from the plan (16.807 GB per token in total for this file). On the reference
laptop `membw` was 30.06 GB/s and `diskbw` 3.30 GB/s ([doctor_maneesh-msi.txt](docs/data/doctor_maneesh-msi.txt));
0.053 s is the measured non-matvec work of a token ([nonmatvec_profile.txt](docs/data/nonmatvec_profile.txt));
the MTP draft step and the marginal verify row were fitted on the ladder
([spec_cost_model.txt](docs/data/spec_cost_model.txt)).

Measured over predicted, plain: 1.02 / 1.13 / 1.51 / 1.91 at 5 / 11 / 16 GiB / resident, the last two hot
([ladder.txt](docs/data/ladder.txt)); 0.98 / 0.98 / 1.00 / 1.06 / 1.22 at 6 / 8 / 12 / 16 / 32 GiB cool
([ladder_phase4.txt](docs/data/ladder_phase4.txt)). The model is right to a few percent where the disk
dominates and optimistic where RAM does, because it assumes the kernels reach the full memory bandwidth and
they reach 74 to 77 % of it. The round model is pessimistic by 10 to 25 % on streamed budgets (0.75 / 0.81 /
0.91 / 0.89, [spec_cost_model.txt](docs/data/spec_cost_model.txt)) because the extra rows' compute hides under
the disk read there. The doctor's own run on the reference laptop is filed as
[doctor_phase6_maneesh-msi.txt](docs/data/doctor_phase6_maneesh-msi.txt).

## Limitations and misses

Every item below is documented in a numbered finding of [docs/payload-vs-doc.md](docs/payload-vs-doc.md)
(76 of them) or a phase report under [docs/](docs/); the finding numbers are given.

1. **Prompt processing is slow: about 3 tokens per second** on this AVX2-only CPU (1.5 to 1.8 x what it was
   before the blocked kernel; 2.4 / 3.0 / 3.0 tok/s at 32 / 128 / 512-token prompts, [prefill.txt](docs/data/prefill.txt),
   finding 73). A 500-token prompt takes nearly three minutes before the first token. The probe in
   [kernel_probe.txt](docs/data/kernel_probe.txt) (finding 70) says why the batched kernel cannot fix it: a Q4_K
   super-block costs 50 cycles on this core, of which the header (18), the scale shuffles (6) and the nibble
   unpack (2) can be shared across activation rows, while the `maddubs -> madd -> add` core (21), the f32 tail
   (12) and the min term (8) are paid per row; the core retires about two uops per cycle on this stream
   whatever its order. What would fix it: AVX-512 VNNI (`vpdpbusd` folds each triple into one instruction,
   taking the unshareable 41 cycles to about 15) or a GPU. Neither is in this release.
2. **`--spec` is a disk-regime feature.** It wins where the disk is the bottleneck (2.27 x at 5 GiB, 1.78 x at
   11 GiB) and is break-even where the model fits (0.99 x resident, 1.17 x at 16 GiB hot;
   [ladder.txt](docs/data/ladder.txt), findings 68, 74, 76), because a verify row costs 0.4 to 0.5 s of compute
   when nothing hides it. Plain decoding is the default; `--spec 3` is what the doctor recommends when it
   predicts a win. On prose the head is accepted less often (25 to 70 % against 90 % on code and lists,
   [spec_acceptance.txt](docs/data/spec_acceptance.txt), finding 67), and on the 8 GiB rung `--spec 4` now edges
   `--spec 3` (2.31 x against 2.23 x, [spec_identity_8g.txt](docs/data/spec_identity_8g.txt), finding 75); k stays
   3 because it loses least where nothing hides the rows.
3. **Every number was measured on one throttling laptop.** Compute speed swings up to 2.5 x with temperature
   while memory bandwidth does not (finding 40, [kernels_bench.txt](docs/data/kernels_bench.txt)). The plain
   resident token is 0.717 s cool ([phase3_timing.txt](docs/data/phase3_timing.txt)) and 1.170 s hot
   ([ladder.txt](docs/data/ladder.txt)); the 16 GiB rung 1.155 cool and 1.572 hot. The table above says which is
   which. The 8 GB and 16 GB rows are memory caps on a 32 GB machine; nobody has yet run this on a real 8 GB
   or 16 GB laptop, which was named a risk before the first line was written ([ARCHITECTURE.md](ARCHITECTURE.md)).
4. **Only one GGUF file is supported**, bartowski's `Qwen3.8-27B-Q4_K_M.gguf` at 17,772,537,440 bytes, sha256
   `e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b`. It has five tensor types (Q4_K, Q5_K,
   Q6_K, Q8_0, Q4_0); the Unsloth "UD" Q4_K_M has nine (IQ4_XS, IQ4_NL, IQ3_S, Q3_K among them) and this engine
   has no kernels for the IQ types, so it refuses the file ([q4km_type_breakdown.txt](docs/data/q4km_type_breakdown.txt),
   findings 1, 22). Other sizes of the same recipe (Q4_K_S, Q5_K_M, Q6_K, Q8_0) share the types but were never
   loaded or tested.
5. **"Identical tokens at every budget" is a statement about one file, not about the model.** Two Q4_K_M
   recipes of the same checkpoint, run by the same engine (llama.cpp), agree on the first four tokens of a
   prose prompt and diverge at the fifth ("rivers plunge over cliffs" against "glaciers once shaped the
   landscape"), with raw logits up to 0.95 apart ([llamacpp_bartowski_vs_ud.txt](docs/data/llamacpp_bartowski_vs_ud.txt),
   finding 24). Between this engine and llama.cpp on the *same* file, greedy tokens match 16/16, 6/16 and
   5/16 on the three test prompts, the splits sitting at margins of 0.02 to 0.10 logits, inside the 0.3 to 0.4
   two 8-bit engines differ by ([phase3_timing.txt](docs/data/phase3_timing.txt), findings 42, 50), and against
   the official b10827 build over 32 tokens 32/32, 6/32 and 32/32 as shipped but 32/32, 6/32 and 5/32 with
   `--no-repack`, llama.cpp's own two kernel paths parting at token 5 of the prose prompt
   ([vs_llamacpp.txt](docs/data/vs_llamacpp.txt)); and the Q4_K_M
   file itself is 0.5 to 0.7 raw logits from the bf16 model ([quant_noise_floor.txt](docs/data/quant_noise_floor.txt),
   finding 35). Token-for-token equality with another engine is a coin toss past the first few tokens of prose;
   argmax and top-10 agreement is what was gated.
6. **The Q4_0 and Q8_0 kernels are compute-bound** (26 to 30 % and 52 to 63 % of memory bandwidth against 68
   to 89 % for the K-quants, [kernels_bench.txt](docs/data/kernels_bench.txt), findings 49, 51), and they have no
   blocked (batched) form, so prefill and verify rows through them take the per-row loop. They are 5.8 % of this
   file (`ssm_out` in half the DeltaNet layers, the MTP block), which is why it was never worth fixing.
7. **The Linux path compiles and has not been run by its author**: `O_DIRECT`, the sysfs sector and disk-type
   queries, large pages through `madvise`, the Ctrl-C handler, `scripts/ladder.sh`. CI now runs the whole
   non-model test suite on `ubuntu-latest`, including the streaming tests on the tiny model, which is that
   path's first execution; its result is on the [Actions page](https://github.com/ManeeshJupalle/Qwen-3.8-in-Rust/actions).
8. **Large pages fall back and are unmeasured.** `SeLockMemoryPrivilege` is not held on the development
   machine, so every arena is on 4 KiB pages; the engine logs the fallback once and continues. Whether 2 MiB
   pages change anything here was never measured (Phase 4 report).
9. **No vision.** The checkpoint's vision tower ships as a separate `mmproj` file that this engine does not
   read; text only (finding 5).
10. **No tool-call template fixtures.** The chat template renders byte-for-byte against HF on seven cases
    (multi-turn, reasoning content, an empty system message, CJK and emoji; finding 66), and `tojson` is
    implemented, but no case with a `tools` list is filed, so tool-call rendering is untested.
11. **Two bugs discarded good measurements, and what they taught.** (a) Windows' performance counters on this
    box report cumulative CPU time as a rate, so a phase concluded a runaway service was eating 5 to 11 cores
    and threw away a day of throughput results that were correct (finding 47); the rule since: measure other
    load with `GetProcessTimes` deltas, and when an annotation contradicts a best-ever measurement, disbelieve
    the annotation. (b) Two PowerShell facts broke the first two ladder runs silently: a process `ExitCode`
    reads as null unless the handle was touched while the process lived, and a `[string]` parameter shares its
    case-insensitive name with a script variable, so the identity gate compared ids against an empty list and
    printed "DIFFER" for ids that matched (finding 61). The ladder runner now re-renders its tables from the
    saved stats so a rendering bug never costs a re-measurement.
12. **The projection versus what landed.** Before the first line, [ARCHITECTURE.md](ARCHITECTURE.md) projected
    1 to 2 tokens per second on a 16 GB machine without a GPU (0.5 to 1 s/token, with MTP) and a cost model of
    20 ms per GB of RAM. What landed: 0.63 tok/s at a 16 GB laptop's free RAM with `--spec 3` (1.58 s/token) and
    1.2 tok/s fully resident on a cool machine (0.83 s), against 5.3 s/token for the official llama.cpp AVX2 build
    on the same CPU, resident (`llama-bench` tg128; [vs_llamacpp.txt](docs/data/vs_llamacpp.txt), finding 21, and
    "Versus llama.cpp" above). The RAM term was 10 x
    optimistic (a 2667 MT/s laptop bus moves 30 GB/s, and 16.8 GB pass through it per token), and the per-row
    cost of verification (item 1) took most of what MTP was expected to give.
13. **Context is what `--max-pos` sizes at load** (4096 in `chat`, 32 KiB of KV per position across the 16
    attention layers), not the model's 262K; the chat refuses a turn that would not fit and says so. Nothing
    past a few hundred positions was measured; attention over a long context runs the per-head scalar path.
    The other side of the same fact: `aqueduct run` sizes the KV cache to the prompt plus `--max-tokens`, so a
    two-token run fits a 3 GiB budget that `chat` refuses, with all 64 layers streamed at 4.8 s/token
    ([clean-clone run](docs/data/clean_clone_maneesh-msi.txt)).
14. **The cost model is optimistic where RAM dominates** (1.2 x cool, 1.9 x hot at resident, finding 55): it
    assumes the kernels stream at the full memory bandwidth and they reach 74 to 77 %. The doctor's number
    for a machine whose model fits is a floor, not an estimate.
15. **The always-resident set costs about a layer more than it needs to.** The 256 MiB baseline reserve is
    generous by roughly 0.2 GB (measured overhead 35 to 60 MB, finding 59), the embedding table (0.72 GB) is
    pinned although one row is read per token, and the MTP block (0.24 GB) is pinned even without `--spec`.
    Each of those is a pinned layer at the margin, 0.25 GB per token from disk, about 0.08 s on the NVMe.
16. **The MTP head in this file is its noisiest copy.** bartowski stores the draft block as Q4_0 (9 % relative
    RMS error against the Q6_K/Q8_0 copies in the Unsloth files, finding 23); how much of the prose acceptance
    gap that costs was not separated from the task itself.
17. **Windows measurement tooling only.** The memory cap (`--job-limit`) is a Windows job object the engine
    puts itself into; the Linux runner (`systemd-run`) is unrun. On Windows the cap is cooperative in the
    sense that the engine applies it, not the launcher.
18. **Sampling has no repetition penalty**; a `generation_config.json` that names one is refused rather than
    ignored. Temperature, top-k, top-p, min-p and a seed are what exists.
19. **The tokenizer, chat template and generation defaults are separate downloads**, not embedded; the engine
    reads the Hugging Face `tokenizer.json` with the `tokenizers` crate (parity 45/45 cases,
    [gguf_vs_hf_tokenizer.txt](docs/data/gguf_vs_hf_tokenizer.txt)) rather than the vocabulary inside the GGUF.
20. **Unbuffered I/O has requirements.** The model must sit on a volume that accepts `FILE_FLAG_NO_BUFFERING`
    / `O_DIRECT` with 4 KiB alignment (local NTFS and ext4 do; network shares and some FUSE filesystems do
    not, and that failure mode was never exercised). The doctor's read benchmark uses the same path, so if it
    runs, the engine will.
21. **The shipped binaries were not measured.** They are compiled for x86-64-v3 so that no AVX2 code runs
    before the CPU check; the ladder was measured with a baseline build whose kernels select AVX2 at run time.
    The kernels are identical hand-written intrinsics in both; the surrounding scalar code differs in codegen.

## Not yet

- **GPU tier.** The design has a tier 0 for the first N layers in VRAM; a 4 to 6 GB NVIDIA card was the
  plan for turning the 16 GB laptop into ~10 tok/s. It would also be the fix for item 1: a prefill is
  compute-bound, so streaming the weights *through* VRAM for the batched pass (even weights that live on the
  disk) would cut the per-row cost that AVX2 cannot. Not started.
- **VNNI / AVX-512 kernels** (`vpdpbusd`): the other fix for item 1; there is no such CPU here to measure on.
- **Unsloth "UD" and other IQ quantisations**: no IQ4_XS / IQ4_NL / IQ3_S kernels.
- **DFlash2** (a 7-token external drafter the vLLM recipe uses): only the in-checkpoint MTP head is used.
- **macOS**, ARM / NEON, a static (musl) Linux binary, an HTTP server, batched serving.

## Validation

Payloads before structs: the first phase downloaded the checkpoint's files and wrote down every place the
documentation and the bytes disagreed (26 items before a kernel existed, 76 now; [docs/payload-vs-doc.md](docs/payload-vs-doc.md)).
The engine's numbers came in order: a tiny 9-layer model of the same graph, built through llama.cpp's real
converter so that the converter's re-tiling and `+1` norms are exercised, decoded 60/60 greedy tokens against
a PyTorch reference ([phase2a_tests.txt](docs/data/phase2a_tests.txt)); every kernel has fixtures with hostile
inputs designed to fail a specific wrong implementation; all 64 real layers were checked against the
dequantised-GGUF reference in f32 mode within a derived budget and in the production Q8 mode on argmax and
top-10 ([phase2b_parity_0_63.log](docs/data/phase2b_parity_0_63.log)); the noise floor between the bf16 model,
the Q4_K_M file and two 8-bit engines was measured before "matching llama.cpp" was interpreted
([quant_noise_floor.txt](docs/data/quant_noise_floor.txt)); and identity gates hold at every thread count, every
budget and every `k`, with the batched prefill and verify passes bit-identical to the token-by-token feed
([phase55_e2e.txt](docs/data/phase55_e2e.txt), [spec_acceptance.txt](docs/data/spec_acceptance.txt)). The decode
step allocates nothing (a counting allocator proves it), so a run that fits its plan cannot run out of memory
later.

## License and credits

The code is dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). The model weights are Qwen's, under the Apache-2.0 license of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); the GGUF quantisation is
[bartowski's](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF), made with llama.cpp and an importance matrix.

- [Qwen](https://huggingface.co/Qwen) for the model, the tokenizer and the chat template.
- [bartowski](https://huggingface.co/bartowski) for the GGUF.
- [ggml](https://github.com/ggml-org/ggml) for the K-quant kernel structure the AVX2 kernels follow.
- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) for the tier design: layer-contiguous
  arenas, a pinned prefix with a prefetching ring, the plan-before-allocate rule, the `--ids` test channel.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) as the reference engine, converter and quantiser.
