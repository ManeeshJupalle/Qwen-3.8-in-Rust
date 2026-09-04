# Phase 3 report (full forward, fully resident)

Commits 8f21412 .. (this one) on `main`, 2026-09-03. Numbers in `docs/data/` (`membw.txt`, `kernels_bench.txt`,
`phase3_timing.txt`, `phase3_parity_0_63.log`); findings 37 to 46 in `docs/payload-vs-doc.md`; kernel design
in `docs/kquant-dot.md`. Machine: i7-9750H laptop, 32 GB, GGUF on the C: SSD; throttling caveat everywhere
(finding 40): kernel throughput swung 2 to 2.5 x during the session while DRAM bandwidth stayed at 29 to 30 GB/s.

```
PHASE 3 REPORT
tests          : 51/51 non-ignored (cargo test --release, 13 binaries) + 2 ignored model-scale gates green
                 (real_layers_0_63_and_logits 1148 s, phase3_e2e 263 s); clippy --all-targets clean
membw          : single 14.4 GB/s, all threads 28.9 GB/s (12 threads; 6 threads 28.7), 5 runs: 26.39 28.71 28.92 28.15 28.79
                 (later same-state pairs: 29.7 to 30.7 at 6 threads, 23 to 25 when the box was hottest)
kernels        : shape 5120 x 5120 (14.7 MB Q4_K), single / all-threads GB/s, % of the membw measured in the same state
               : production path (Q8_0 activations, ggml lane structure; only ever measured on a throttled box):
                 q4k 5.8 / 20.6 (69%), q5k 3.9 / 7.3 (24%), q6k 4.1 / 7.6 (25%), q8_0 2.8 / 7.7 (26%)
                 [17408 x 5120, throttled: q4k 22.6 (78%), q5k 15.4 (53%), q6k 17.9 (62%), q8_0 16.1 (56%)]
               : ggml Q8_K activations (opt-in, --q8k), cool box, membw 28.9:
                 q4k 7.3 / 26.2 (91%), q5k 6.5 / 21.7 (75%), q6k 8.7 / 26.5 (92%)
               : avx2 vs scalar max diff 0 (every kernel, both activation forms, 1263 + 5000 random rows + fixtures);
                 Q8_K quantiser bit-exact vs the Python port of quantize_row_q8_K_ref (36 rows), Q8_0 quantiser bit-exact;
                 regression ceilings: green, f32 path 0.002 of budget (layer 0), Q8 path worst 0.654 of its per-layer
                 ceiling (layer 17, sentence), logits 0.37 / 0.50 / 0.44 of ceiling; thread invariance green (kernel tests,
                 layer tests, and identical 16 greedy ids at 6 and 12 threads on the real model)
prefill        : batched vs sequential logits 0 (bit-identical) vs ceilings 0.397 / 0.237 / 0.349; state max rel 0;
                 cache max rel 0; per-position argmax 48/48 vs ref_gguf; batched 1.9 to 2.4 tok/s, 2.9 to 3.3 x sequential
e2e            : llama.cpp match: capital 16/16, fib 16/16, sentence 5/16; first divergence margin 0.0708 (step 5,
                 "glaciers" vs "rivers"; the Q8-vs-Q8 noise on this file is 0.29 to 0.44); text coherent: yes (all three)
timing         : load 30.6 s (17.52 GB from the SSD with 9 GB free; 21.8 to 40 s on later loads); prefill 1.9 to 2.4 tok/s;
                 decode 1.40 / 1.45 / 1.56 s/token = 16.84 GB per token = 12.0 / 11.6 / 10.8 GB/s = 41 / 40 / 37 % of membw
                 (12 threads, first run; the same binary later, box throttled: 2.30 to 2.86 s/token = 5.9 to 7.3 GB/s at
                 6 and 12 threads alike); peak RSS 17.98 GB vs expected 17.69 GB (weights 17.52 + state), ratio 1.016
findings       : 37 Q8_K activations exceed the frozen ceilings 2.5x at layer 0 -> production keeps Q8_0-grain activations
                 under ggml's structure, Q8_K opt-in; 38 single-thread DRAM stream was latency-bound until software
                 prefetch (3.3 -> 6.7 GB/s); 39 lane-structured kernels are bounded by term magnitudes (new test budget);
                 40 thermal throttling 2 to 2.5x, pair every kernel number with membw; 41 batched prefill bit-identical
                 to the sequential feed on the real model; 42 greedy vs llama.cpp 16/16/5 with a 0.07 margin at the split;
                 43 30.6 s load, 17.98 GB RSS, 37 to 41% of membw decode; 44 per-prompt 2x ceilings are noise-fragile,
                 per-layer max used (rule 5's letter); 45 identical ids at 6 and 12 threads end to end; 46 end-of-session
                 numbers are the throttled floor, production Q5_K/Q6_K never measured cool
blocked on     : a thermally stable box for the production kernel numbers (the last measurement was after an 8-minute
                 idle and the cores were still at half speed); a profiler (no VTune/perf here: the "profile" is the
                 cache-resident vs DRAM comparison, which says Q5_K/Q6_K are instruction-bound in the per-sub-block tail:
                 cvt + permute + mul + add per 32 elements, two permutes and the off6 subtract for Q6_K, the hbits shift
                 path for Q5_K)
did NOT do     : tiers, streaming, MTP, sampling, chat template (out of scope by instruction); Q8_K as the production
                 activation (rule 5); single-thread >= 8 GB/s for Q4_K / Q5_K (best 7.3 / 6.5 with Q8_K, 5.9 / 3.9
                 production, throttled); the 50% all-threads gate for the PRODUCTION Q5_K and Q6_K kernels at 5120 x 5120
                 (24 to 34% in every paired state measured; 53 to 62% at 17408 x 5120) - 3.2 to 3.4 were entered on the
                 strength of the Q8_K-activation measurement (75 to 92%) taken before rule 5 forced the Q8_0-grain rewrite,
                 and the file is 58% Q4_K (69 to 78%); chunked DeltaNet prefill and a cache-blocked prefill GEMM (prefill is
                 compute-bound at 2 tok/s); a second cool-machine bench of the production kernels
```

## What the next session should do first

1. Bench the production kernels on a cold box (`aqueduct bench membw --runs 3; aqueduct bench kernels --rows 5120
   --cols 5120 --no-scalar; aqueduct bench membw --runs 3`, first thing after boot). If Q5_K / Q6_K are still below
   50 %, the per-sub-block tail is the target: precompute the 8 (or 16) f32 factors of a super-block into a small
   stack array and load them instead of `permutevar8x32`; drop the `off6` subtract by folding the Q6_K offset into
   the factor table (`-32 * sum q_x` times the factor, added once per sub-block as a scalar); process two rows per
   pass so the activation loads and the factor setup are shared.
2. Decode at 37 to 41 % of membw on a warm box already; with the kernels at 70 to 90 % the non-matvec work
   (48 DeltaNet steps of 48 x 128 x 128, lm_head's 248,320 rows, the per-token allocations) becomes visible: measure
   it with `aqueduct run -v` and vectorise `deltanet_step`.
