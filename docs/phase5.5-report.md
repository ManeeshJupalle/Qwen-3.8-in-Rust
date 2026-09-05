# Phase 5.5 report (the blocked GEMM)

Commits 9b55bb9 (the kernel, tests, bench, probe), 0e9716e (the tile sweep, the end-to-end identity run, findings
70 to 72) and the ones after them on `main`, 2026-09-05. Design and mechanism in `docs/kquant-dot.md` (the Phase 5.5
section); numbers in `docs/data/` (`kernel_probe.txt`, `matmul_t_bench.txt`, `phase55_e2e.txt`, `prefill.txt`,
`spec_acceptance.txt`, `spec_identity_8g.txt`, `ladder.txt`, `spec_cost_model.txt`) and `docs/ladder.md`; findings
70 to TODO_LASTFINDING in `docs/payload-vs-doc.md`. The Phase 5 versions of the refiled files are kept as
`docs/ladder_phase5.md`, `docs/data/ladder_phase5.txt`, `spec_acceptance_phase5.txt`, `spec_identity_8g_phase5.txt`,
`spec_cost_model_phase5.txt`. Machine as in Phases 4 and 5 (i7-9750H, 6 cores / 12 threads, L1D 32 KB and L2 256 KB
per core, L3 12 MB, 32 GB DDR4-2667, NVMe, Windows 11), through this session in its **warm** state: the frequency
counter read 3.1 GHz all-core under the kernels, membw 25 to 28 GB/s, and the plain resident token measured 0.93 s
against 0.72 s in the cool Phase 3.6 run. Every speed-up below is an A/B taken in the same block of minutes
(the two variants alternated or run back to back), and the identity gates do not depend on the clock at all.

```
PHASE 5.5 REPORT
tests          : 78/78 non-ignored (22 unit + 56 integration over 23 test targets: Phase 5's 74 + matmul_t 3 (scalar blocked ==
                 single-row scalar bit for bit and within the terms budget on 441 fixture dots; AVX2 blocked == scalar blocked == the
                 single-row AVX2 kernel on 5028 fixture / random / hostile dots for every tile 1..16; matmul / matmul_fn / matmul_t ==
                 matvec for tiles 0..16, batches 1..35, 1..12 threads, both shapes) + matmul_t_alloc 1 (the blocked matmul_fn allocates
                 nothing, tiles 1/4/8/16, batches 1/4/35, 1 and 6 threads)) + 2 ignored model-scale gates unchanged; clippy --all-targets clean
kernel         : dot_q8k_t (kdot.rs scalar reference, avx2.rs) = one K-quant row against a tile of <= 16 Q8_K rows, each super-block
                 unpacked once (Q4_K/Q5_K: 8 code vectors + 8 pre-shuffled scale broadcasts + the 8 mins; Q6_K: 8 code vectors + 8 scale
                 pairs), sub-blocks outer and activations inner in register groups of 8/4/2/1; per row the single-row kernel's exact i32
                 lanes, its f32 multiply-then-add per super-block and its reduction, so out[i] == dot_q8k(row, xs[i]) bit for bit (a property
                 the tests check; the contract stays the frozen Q8_K ceilings). matvec::matmul_t drives it: row blocks of 32, every tile
                 applied to a block before the next (a weight byte leaves DRAM once per batch), the tile gathered into a stack array
                 (allocation-free). matmul and matmul_fn route K-quant weights x Q8_K rows here; F32 / Q8_0 / Q4_0 weights and --q8-fine
                 rows keep the per-row loop. Batched matmuls (>= 2 rows) run on every hardware thread, the single-row matvec on the physical
                 cores (finding 71; AQUEDUCT_MATMUL_THREADS pins, AQUEDUCT_MATMUL_T=0 is the Phase 3.3 loop for A/B).
tile           : T in {2, 4, 8, 16} measured on 17408 x 5120 (gate) and 5120 x 17408 (down) at 32 and 4 activations, 12 threads, tiles
                 alternated per repetition (docs/data/matmul_t_bench.txt, finding 72). ms per activation row at n = 32, per-row loop ->
                 tile 16: Q4_K 1.337 -> 1.140 (1.17 x), Q5_K 1.535 -> 1.153 (1.33 x), Q6_K 1.686 -> 0.993 (1.70 x) on gate; on down
                 tile 8 beats 16 for Q5_K / Q6_K (16 rows of 21 KB overflow the 256 KB L2), so the tile is capped to the largest power of
                 two whose rows fit 192 KB: 16 at the hidden width, 8 at the intermediate width. A tile of 2 always loses. At n = 4 (the
                 --spec 3 verify batch) Q4_K is at par with the per-row loop (0.87 to 0.96 x) and Q6_K gains 1.3 to 1.7 x.
why not 3 x    : tools/probe, docs/data/kernel_probe.txt, finding 70. The single-row Q4_K kernel costs 50 cycles per super-block with
                 everything in L1 (2.5 x the port-model floor); the parts a tile can share are the header (18), the scale shuffles (6) and
                 the nibble unpack (2); the maddubs -> madd -> add core (21: a triple costs 1.4 cycles from registers, 2.3 with its load,
                 against 1.0 by the tables; alignment irrelevant), the f32 tail (12) and the min term (8) are per activation. Floor at tile
                 8: 44 cycles against 50. Software pipelining, split accumulators, ggml's scalar utmp unpack and the JCC-erratum padding
                 change nothing (48 to 55). The sibling hardware thread does: 12 threads over 6 give 1.28 / 1.36 / 1.52 x (Q4_K / Q5_K /
                 Q6_K) on the batched kernel because the ports sit 60 % idle behind the chains (finding 71).
identity       : batched prefill vs token-by-token feed on the real model: hidden max|diff| 0, last logits max|diff| 0, carried state and
                 KV cache max rel 0 on all 3 fixture prompts; argmax per prompt position 5/5, 4/4, 37/39 (the two flips inside the noise
                 floor, as Phase 3.5); llama.cpp first-16 match 16/16, 6/16, 5/16 as Phase 3; the 96 greedy ids equal
                 tests/fixtures/ladder_expected_ids.json (docs/data/phase55_e2e.txt); prefill.txt: the first generated token identical
                 across the per-row and blocked configurations at every prompt length and budget.
                 spec vs no-spec 200-token ids: TODO_IDENTITY
prefill        : GATE NOT MET (>= 3 x Phase 3's asked; 1.5 to 1.8 x measured). tok/s on 32 / 128 / 512-token prompts (prefixes of one
                 text, docs/data/prefill.txt; --max-tokens 1; same-session A/B, the three configurations back to back per prompt):
                   resident  Phase 3 path (per-row loop, 6 thr) 1.55 / 1.68 / 1.67   blocked 6 thr 1.84 / 2.19 / 2.20   blocked 12 thr 2.41 / 2.97 / 3.04
                             speed-up over the Phase 3 path 1.56 / 1.77 / 1.82 x (blocking alone 1.19 / 1.31 / 1.32 x, the sibling thread the rest)
                   11 GiB    Phase 3 path 1.81 / 1.83 / 1.74   blocked 6 thr 2.11 / 2.19 / 2.23   blocked 12 thr 2.74 / 3.03 / 3.07   speed-up 1.52 / 1.66 / 1.77 x
                 (35 or 36 layers pinned, 28 or 29 streamed once per prompt; the 11 GiB runs came later in a cooler state, membw 26.7 -> 29.2 GB/s
                 across the step, so their absolute numbers sit above the resident ones; each speed-up pairs runs minutes apart)
                 per token at 512: 0.60 s (Phase 3 path) -> 0.33 s (blocked, 12 thr); the fixture prompts in phase55_e2e.txt: 1.98 / 1.89 / 2.75 tok/s
                 for 5 / 4 / 39 tokens against Phase 3.6's cool 1.95 / 1.98 / 2.13 with a decode token 1.29 x slower this session (0.926 vs 0.717 s)
c_verify       : TODO_CVERIFY
ladder         : TODO_LADDER
default        : TODO_DEFAULT
findings       : 70 an extra activation row is the int8 multiply-accumulate, not the unpack (26 of 50 cycles shareable); 71 the batched
                 kernels leave the ports idle and the sibling thread fills them (12 threads for batches, 6 for the matvec); 72 the tile:
                 16 at hidden width, 8 at intermediate width, 2 always loses, at 4 rows Q4_K gains nothing; TODO_FINDINGS
blocked on     : nothing
did NOT do     : TODO_DIDNOT
```

TODO_BODY
