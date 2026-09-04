# Phase 3.5 report (kernel decision)

Commits 73b1850 .. (this one) on `main`, 2026-09-04. Numbers in `docs/data/` (`bench_rounds.log`,
`kernels_bench.txt`, `phase3_timing.txt`, `two_row_ab.log`, `phase3_5_single_thread.log`,
`phase3_5_parity_q8k_0_63.log`); findings 47 to 51 in `docs/payload-vs-doc.md`; kernel design and the decision
in `docs/kquant-dot.md`; protocol `tools/bench_rounds.ps1`, ceilings freezer `tools/freeze_ceilings.py`.
Machine: i7-9750H laptop (6 cores / 12 threads), 32 GB DDR4-2667, GGUF on the C: NVMe SSD.

**Correction to the first draft of this report.** It filed the whole throughput half as unusable, claiming a
runaway `bthserv` service held 9 of 12 cores. That service was an artifact of the Windows performance
counters, which on this box report a process's CPU time accumulated since boot as though it were a rate
(finding 47); `GetProcessTimes` measures the same process at 0.00 cpu-s per 4 s. The machine was idle
throughout. Everything below is re-measured with the load accounted for correctly (other load 0.09 to 0.23 of
12 cores during every step), and the conclusions reverse: the kernel gate is met and the kernels are
memory-bound.

```
PHASE 3.5 REPORT
decision       : Q8_K promoted: YES. Q8_K-path logits vs ref_gguf (capital / fib / sentence): max|diff| 0.362 / 0.281 / 0.357
                 (criterion: below the bf16 noise floor 0.653 / 0.516 / 0.679), argmax 3/3, top-10 10 / 10 / 9 (>= 9/10);
                 vs llama.cpp (Q8_K itself) 0.349 / 0.226 / 0.392, argmax 3/3, top-10 9 / 10 / 9. Q8_K is the production
                 activation for K-quant weights; the Q8_0-grain path is `aqueduct run --q8-fine`. It also costs nothing in
                 speed: faster than Q8_0-grain for Q5_K and Q6_K in 3 rounds of 3, a wash for Q4_K. Outside the criterion:
                 argmax per prompt position 5/5, 4/4, 37/39 (Q8_0: 48/48; both flips inside the noise floor, printed by the gate).
rule 5         : amended: ceilings are per activation format, each frozen from its own 0..63 measurement at 2 x the per-layer
                 max over prompts (tests/fixtures/q8k_ceilings.json from docs/data/phase3_5_parity_q8k_0_63.log, 846 s;
                 q8_ceilings.json unchanged); AQUEDUCT_ACT=q8k|q8_0 selects the format in real_layers.rs / phase3_e2e.rs,
                 a format with no frozen file runs ungated as its measurement (tools/freeze_ceilings.py freezes the log)
threads        : default is now the PHYSICAL core count (crates/core/src/cpu.rs: GetLogicalProcessorInformationEx on Windows,
                 thread_siblings_list under /sys on Linux, available_parallelism as fallback); --threads N overrides.
                 6 beats 12 on 35 of 36 K-quant kernel lines (one tie) and on membw itself, and by 25 % end to end.
tests          : 52/52 non-ignored (cargo test --release, 15 test targets + doc-tests) + 2 ignored model-scale gates green
                 (real_layers_0_63_and_logits in q8k mode 846 s = the measurement; phase3_e2e 216 s on the production config);
                 real_layers_0_7 green under both formats (q8_0 at 0.634 of its ceiling, unchanged from Phase 3: the
                 kernels are bit-identical); clippy --all-targets clean
bench protocol : plugged in, 5 min idle, 3 rounds of membw then kernels back to back (5120 x 5120 and 17408 x 5120), every
                 round filed, not best-of; each step bracketed by a GetProcessTimes-delta measurement of other load
                 (0.09 to 0.23 of 12 cores throughout). Performance counters are not used at all (finding 47).
membw          : per round, best of 3: 1 thr 14.66 / 17.17 / 16.83; 6 thr 29.13 / 32.31 / 31.59; 12 thr 26.77 / 32.28 / 30.59 GB/s
kernels        : production (Q8_K rows), 6 threads, % of that round's own membw, rounds 1 / 2 / 3:
                 5120 x 5120:  q4k 68 / 77 / 70 %, q5k 74 / 75 / 79 %, q6k 85 / 85 / 84 %   -> the 50 % gate: MET with room
                 17408 x 5120: q4k 83 / 75 / 78 %, q5k 85 / 76 / 77 %, q6k 89 / 81 / 81 %
               : --q8-fine (Q8_0-grain rows), 6 threads, 5120 x 5120: q4k 74 / 71 / 72 %, q5k 67 / 67 / 68 %, q6k 83 / 75 / 78 %
               : 12 threads is worse on every K-quant line but one tie; Q4_0 (5.8 % of the file with Q8_0 weights) prefers 12
               : the fix: two rows per pass built for both formats, bit-identical, measured slower on 51 of 54 A/B lines
                 (ratio 0.65 to 1.16, mostly 0.85 to 0.95) and REMOVED -- the right answer for a memory-bound kernel;
                 factor tables (Q4_K / Q6_K Q8_0-grain), two-mask Q6_K unpack and the fixed Q5_K shift (Q8_K kernel) kept,
                 bit-identical, inside this box's noise except the Q8_K Q5_K kernel (7.1 to 7.5 vs 4.0 to 4.5 GB/s at 1 thread);
                 the Q8_0-grain Q5_K kernel keeps its Phase 3 body (the rewrite measured 30 % slower for it)
parity (q8k)   : per-layer relative error 6.7e-4 to 2.2e-3 at layer 0, worst 5.3e-2 at layer 63 sentence (Q8_0: 1.6e-2);
                 2 to 4 x the Q8_0 path, as finding 31 predicted for one scale per 256 vs per 32; incremental bit-exact;
                 f32 mode 0.002 of budget (unchanged)
prefill        : batched vs sequential logits 0 (bit-identical), state / cache max rel 0; 1.7 to 2.0 tok/s
e2e            : llama.cpp match: capital 16/16 (Phase 3 text verbatim), fib 6/16 (Phase 3: 16/16; step 6 margin 0.095, ours
                 " 1" vs " 0", continuation `if n <= 1: return n ... fibonacci(n-1) + fibonacci(n-2)`), sentence 5/16
                 (margin 0.024 at the split, as before); text coherent: yes (all three)
timing         : load 22.4 s; decode 1.151 / 1.154 / 1.178 s/token = 14.60 / 14.56 / 14.27 GB/s = 45 to 46 % of membw 31.6
                 (6 threads; 12 threads: 1.443 / 1.454 / 1.446 = 11.6 GB/s = 37 %); 1.22 to 1.36 x Phase 3's per-token best;
                 peak RSS 17.975 GB vs 17.686 expected, ratio 1.016
                 -> the >= 60 % end-to-end target: NOT met. Not the kernels (75 to 89 % of membw): at the file's type mix the
                 matvecs account for ~0.69 s of the 1.15 s token; the other ~0.46 s (40 %) is DeltaNet, conv, norms, allocations
findings       : 47 the perf counters report cumulative CPU as a rate and invented a runaway service, which made the first
                 draft discard its own good results; measure other load with GetProcessTimes deltas; 48 Q8_K meets the
                 criterion, rule 5 per activation format, the two `sentence` flips are inside the noise floor and the gate
                 bounds them; 49 the 50 % kernel gate is met (68 to 89 %), the two-row pass is slower and was dropped;
                 50 e2e 1.15 s/token at 46 % of membw, 40 % of the token is non-matvec; 51 six threads beat twelve
                 everywhere that matters, so the default is physical cores
blocked on     : nothing
did NOT do     : reach the 60 % end-to-end target (45 to 46 %; the remaining work is non-matvec, not kernels); ship the
                 two-row pass (measured slower); vectorise deltanet_step or remove the per-token allocations; explain why the
                 same shift rewrite helps the Q8_K Q5_K kernel and hurts the Q8_0-grain one, or the ~30 % second-position
                 effect on the Q5_K matrix at one thread (no profiler here); tiers, streaming, MTP, sampling, chat template
                 (out of scope by instruction)
```

## What the next session should do first

1. The token is 1.15 s and about 0.46 s of it is not matvec. Measure that directly (`aqueduct run -v` gives
   per-token wall time; add a per-stage timer behind a flag if it is still not obvious) and then take the two
   obvious pieces: vectorise `deltanet_step` (48 heads x 128 x 128 f32 per layer, 48 layers, still scalar,
   `docs/simd.md`) and hoist the per-token allocations out of `forward_hidden` / `logits_of`. A 60 % token is
   0.89 s, so about 0.26 s has to come out of that 0.46 s.
2. `matmul` (the prefill path) is still single-row and was never re-measured after 3.5; prefill sits at 1.7 to
   2.0 tok/s and is compute-bound. A cache-blocked GEMM is the Phase 3 item that still stands.
3. Phase 4 (tiers) can start on these kernels: they are memory-bound at 75 to 89 % of the bus, which is the
   condition the ladder's cost model assumes.
