# Phase 3.5 report (kernel decision)

Commits 73b1850 .. (this one) on `main`, 2026-09-04. Numbers in `docs/data/` (`bench_rounds.log`,
`two_row_ab.log`, `phase3_5_single_thread.log`, `kernels_bench.txt`, `phase3_timing.txt`,
`phase3_5_parity_q8k_0_63.log`); findings 47 to 50 in `docs/payload-vs-doc.md`; kernel design and the
decision in `docs/kquant-dot.md`; protocol `tools/bench_rounds.ps1`, ceilings freezer
`tools/freeze_ceilings.py`. Machine: i7-9750H laptop, 32 GB, GGUF on the C: SSD. **Machine caveat for
every number below (finding 47): a runaway `bthserv` (Bluetooth Support Service) svchost consumed 5 to 11 of
the 12 hardware threads for the whole session and the ACPI thermal zone read 96 to 97 C throughout; turning
Bluetooth off did not stop it and restarting the service needs an elevated shell. Correctness results are
unaffected; every throughput number is a floor under that load.**

```
PHASE 3.5 REPORT
decision       : Q8_K promoted: YES. Q8_K-path logits vs ref_gguf (capital / fib / sentence): max|diff| 0.362 / 0.281 / 0.357
                 (criterion: below the bf16 noise floor 0.653 / 0.516 / 0.679), argmax 3/3, top-10 10 / 10 / 9 (>= 9/10);
                 vs llama.cpp (Q8_K itself) 0.349 / 0.226 / 0.392, argmax 3/3, top-10 9 / 10 / 9. Q8_K is the production
                 activation for K-quant weights; the Q8_0-grain path is `aqueduct run --q8-fine`. Outside the criterion:
                 argmax per prompt position 5/5, 4/4, 37/39 (Q8_0: 48/48; both flips inside the noise floor, printed by the gate).
rule 5         : amended: ceilings are per activation format, each frozen from its own 0..63 measurement at 2 x the per-layer
                 max over prompts (tests/fixtures/q8k_ceilings.json from docs/data/phase3_5_parity_q8k_0_63.log, 846 s;
                 q8_ceilings.json unchanged); AQUEDUCT_ACT=q8k|q8_0 selects the format in real_layers.rs / phase3_e2e.rs,
                 a format with no frozen file runs ungated as its measurement (tools/freeze_ceilings.py freezes the log)
tests          : 51/51 non-ignored (cargo test --release, 15 test targets + doc-tests) + 2 ignored model-scale gates green
                 (real_layers_0_63_and_logits in q8k mode 846 s = the measurement; phase3_e2e 394 s on the production config);
                 real_layers_0_7 green under both formats (q8_0 at 0.634 of its ceiling, unchanged from Phase 3: the
                 kernels are bit-identical); clippy --all-targets clean
membw          : per round (best, 1 / 6 / 12 threads): 15.5 / 27.3 / 27.2, 16.7 / 31.6 / 28.1, 16.9 / 30.3 / 32.0 GB/s
kernels        : protocol: plugged in, 5 min idle, 3 rounds of membw then kernels back to back (5120 x 5120, the Phase 3
                 binary at 5120 x 5120 in the same state, 17408 x 5120), every round filed with thermal zone / clock / other load
               : production (Q8_K rows), sustained, all-threads best of 6 / 12, % of that round's membw, rounds 1 / 2 / 3:
                 5120 x 5120:  q4k 18.1 / 23.9 / 23.2 GB/s (66 / 76 / 73 %), q5k 6.3 / 7.7 / 8.2 (23 / 25 / 26 %),
                               q6k 7.9 / 8.4 / 11.2 (29 / 27 / 35 %)   -> the 50 % gate for Q5_K / Q6_K: NOT demonstrated
                 17408 x 5120: q4k 20.0 / 18.7 / 23.3 (73 / 59 / 73 %), q5k 12.3 / 13.4 / 15.9 (45 / 42 / 50 %),
                               q6k 13.0 / 15.1 / 17.0 (48 / 48 / 53 %); q8_0 weights (kernel unchanged) 12.3 / 15.4 / 13.0 (45 / 49 / 41 %)
               : --q8-fine (Q8_0-grain rows), 5120 x 5120: q4k 5.7 to 8.0 (20 to 25 %), q5k 6.4 to 9.9 (23 to 31 %), q6k 6.9 to 9.4
                 (22 to 29 %); the Phase 3 binary in the same rounds: q4k 14.4 to 21.9 (45 to 71 %), q5k 7.4 to 9.2, q6k 2.4 to 10.5
               : burst A/B in one process (docs/data/two_row_ab.log), single-row kernels, 17408 x 5120, 6 threads: q4k 33.6,
                 q5k 30.4, q6k 30.5 GB/s (Q8_K rows), 29.1 / 30.5 / 30.0 (Q8_0 rows) = at or above the same minutes' membw
               : single thread, alternating binaries, first position on each matrix: q4k 7.5 to 7.9 (Phase 3 7.5 to 7.7),
                 q5k 7.1 to 7.5 (Phase 3 4.0 to 4.5, always second; 3.5 in second position 5.1), q6k 5.7 to 6.0 (5.4 to 5.9)
               : the fix: two rows per pass built for both forms, bit-identical, measured slower on 51 of 54 A/B lines
                 (ratio 0.65 to 1.16, mostly 0.85 to 0.95) and REMOVED; factor tables (Q4_K / Q6_K Q8_0-grain), two-mask Q6_K
                 unpack and the fixed Q5_K shift (Q8_K kernel) kept, bit-identical; the Q8_0-grain Q5_K kernel keeps its
                 Phase 3 body (the rewrite measured 30 % slower for it in the same position)
               : two measurement effects a reader must know: the first configuration of a sustained run gets the turbo budget
                 (18 to 24 GB/s at 6 threads), everything after it 6 to 11; at one thread whichever activation form runs
                 second on the Q5_K matrix reads ~30 % lower, both binaries, both orders (cause not identified)
parity (q8k)   : per-layer relative error 6.7e-4 to 2.2e-3 at layer 0, worst 5.3e-2 at layer 63 sentence (Q8_0: 1.6e-2);
                 2 to 4 x the Q8_0 path, as finding 31 predicted for one scale per 256 vs per 32; incremental bit-exact;
                 f32 mode 0.002 of budget (unchanged)
prefill        : batched vs sequential logits 0 (bit-identical), state / cache max rel 0, on the production config;
                 1.2 to 1.7 tok/s (Phase 3: 1.9 to 2.4; the machine)
e2e            : llama.cpp match: capital 16/16 (Phase 3 text verbatim), fib 6/16 (Phase 3: 16/16; step 6 margin 0.095, ours
                 " 1" vs " 0", continuation `if n <= 1: return n ... fibonacci(n-1) + fibonacci(n-2)`), sentence 5/16
                 (margin 0.024 at the split, as before); text coherent: yes (all three)
timing         : load 30.8 s; decode 2.51 / 2.24 / 2.25 s/token = 6.7 / 7.5 / 7.5 GB/s = 23 / 26 / 26 % of membw 28.9
                 (12 threads, bthserv at 9 to 10 cores) -> the >= 60 % end-to-end target: NOT met, not judgeable on this box;
                 peak RSS 17.977 GB vs 17.686 expected, ratio 1.016
findings       : 47 runaway bthserv at 5 to 11 cores, Phase 3's "throttling" was in part that, protocol now files the state;
                 48 Q8_K meets the criterion, rule 5 per activation format, the flips at positions 26 / 29 of `sentence` are
                 inside the noise floor and the e2e gate prints and bounds them; 49 two-row slower and dropped, table / shift
                 rewrites bit-identical and unmeasurable here, the sustained bench is the machine; 50 e2e on the production
                 config: same capital text, a different correct Fibonacci, 23 to 26 % of membw under the service
blocked on     : an elevated shell to restart bthserv (or a reboot), then `powershell -File tools/bench_rounds.ps1` and the e2e
                 again: until then neither the 50 % kernel gate nor the 60 % end-to-end target can be judged; the burst A/B
                 says the kernels themselves reach membw
did NOT do     : ship the two-row pass (measured slower); reach the 50 % gate at 5120 x 5120 for Q5_K / Q6_K in a sustained
                 run (23 to 35 %) or the 60 % end-to-end target (23 to 26 %), both under the machine caveat; explain the
                 second-position Q5_K effect or why the same shift rewrite helps the Q8_K Q5_K kernel and hurts the Q8_0-grain
                 one (no profiler here); tiers, streaming, MTP, sampling, chat template (out of scope by instruction)
```

## What the next session should do first

1. Restart `bthserv` from an elevated shell (`Restart-Service bthserv -Force`) or reboot, confirm with
   `Get-CimInstance Win32_PerfFormattedData_PerfProc_Process` that nothing else is above a few percent, and run
   `powershell -ExecutionPolicy Bypass -File tools/bench_rounds.ps1 -Rounds 3` (no `-Baseline` needed) and
   `cargo test --release --test phase3_e2e -- --ignored --nocapture`. Those two files are the Phase 3.5 numbers
   on a healthy box; everything in this report about throughput is provisional until then.
2. If the sustained Q5_K / Q6_K numbers are still below 50 % on a healthy box, the burst A/B pattern
   (`docs/data/two_row_ab.log`, since removed from the bench) is the tool to separate kernel from machine:
   interleave the variants in one process and compare medians.
3. Decode at 60 % of membw needs the non-matvec work measured (`aqueduct run -v` per-token timing) and
   `deltanet_step` vectorised; that was the Phase 3 item 2 and still stands.
