# Phase 5.5 report (the blocked GEMM)

Commits 9b55bb9 (the kernel, tests, bench, probe), 0e9716e (the tile sweep, the end-to-end identity run, findings
70 to 72) and the ones after them on `main`, 2026-09-05. Design and mechanism in `docs/kquant-dot.md` (the Phase 5.5
section); numbers in `docs/data/` (`kernel_probe.txt`, `matmul_t_bench.txt`, `phase55_e2e.txt`, `prefill.txt`,
`spec_acceptance.txt`, `spec_identity_8g.txt`, `ladder.txt`, `spec_cost_model.txt`) and `docs/ladder.md`; findings
70 to 76 in `docs/payload-vs-doc.md`. The Phase 5 versions of the refiled files are kept as
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
                 spec vs no-spec 200-token ids: resident k=1 6/6 k=2 6/6 k=3 6/6 k=4 6/6 k=5 6/6 (all six prompts; docs/data/spec_acceptance.txt);
                 8 GiB k=1 3/3 k=2 3/3 k=3 3/3 k=4 3/3 (200 tokens, 22 pinned / 42 streamed, docs/data/spec_identity_8g.txt); tiny model: the
                 Phase 5 suite unchanged (200 tokens x 3 prompts x k=1..4 resident and streamed, every acceptance count forced, state equal
                 after every round: the tiny model is F32 and never reaches the blocked kernels)
8 GiB          : plain 3.658 s/token; --spec 1..4 1.65 / 2.05 / 2.23 / 2.31 x (Phase 5: 1.66 / 1.98 / 2.08 / 2.03); verify per round 3.94 / 4.08 /
                 4.25 / 4.41 s for 2..5 rows (Phase 5: 3.90 / 4.21 / 4.57 / 5.12): the extra rows cost 0.16 s each under the disk pass, 0.41 before;
                 --spec 4 now overtakes --spec 3 there (finding 75)
acceptance     : resident mean accepted/round at k=1..5: 0.87 1.51 1.98 2.33 2.60 (Phase 5's to the digit: same ids, same text); s/token vs plain
                 1.01 / 1.02 / 0.97 / 0.81 / 0.72 x (Phase 5: 1.06 / 0.98 / 0.94 / 0.86 / 0.76) as the mean over prompts of a sweep through which the
                 plain token heated from 0.75 to 1.50 s; per prompt at k=3 (pairs minutes apart): capital 1.15 fib 1.18 code 1.39 essay 1.07 fact 0.88
                 sentence 0.51 x (finding 74)
prefill        : GATE NOT MET (>= 3 x Phase 3's asked; 1.5 to 1.8 x measured). tok/s on 32 / 128 / 512-token prompts (prefixes of one
                 text, docs/data/prefill.txt; --max-tokens 1; same-session A/B, the three configurations back to back per prompt):
                   resident  Phase 3 path (per-row loop, 6 thr) 1.55 / 1.68 / 1.67   blocked 6 thr 1.84 / 2.19 / 2.20   blocked 12 thr 2.41 / 2.97 / 3.04
                             speed-up over the Phase 3 path 1.56 / 1.77 / 1.82 x (blocking alone 1.19 / 1.31 / 1.32 x, the sibling thread the rest)
                   11 GiB    Phase 3 path 1.81 / 1.83 / 1.74   blocked 6 thr 2.11 / 2.19 / 2.23   blocked 12 thr 2.74 / 3.03 / 3.07   speed-up 1.52 / 1.66 / 1.77 x
                 (35 or 36 layers pinned, 28 or 29 streamed once per prompt; the 11 GiB runs came later in a cooler state, membw 26.7 -> 29.2 GB/s
                 across the step, so their absolute numbers sit above the resident ones; each speed-up pairs runs minutes apart)
                 per token at 512: 0.60 s (Phase 3 path) -> 0.33 s (blocked, 12 thr); the fixture prompts in phase55_e2e.txt: 1.98 / 1.89 / 2.75 tok/s
                 for 5 / 4 / 39 tokens against Phase 3.6's cool 1.95 / 1.98 / 2.13 with a decode token 1.29 x slower this session (0.926 vs 0.717 s)
c_verify       : GATE NOT MET (<= 0.2 s per extra row asked). Fitted as Phase 5 defines it, (verify per round - plain token) / k at the resident
                 rung of the ladder: 0.527 s (Phase 5: 0.558), on a hot rung (plain token 1.170 s against Phase 5's cool 0.830; verify per round
                 2.752 against 2.504). In the token's own units the row fell from 0.67 to 0.45 plain tokens, the kernel's 1.5 x; on the cool
                 prompts of the resident sweep 0.41 to 0.46 s; at 8 GiB the rows add 0.16 s each on top of the disk pass (0.41 in Phase 5). c_mtp
                 0.085 s. docs/data/spec_cost_model.txt (tools/spec_cost_model.py --rungs 5G,11G,16G,resident; the temp directory keeps older
                 runs' stats files, hence the filter)
ladder (spec)  : 5 GiB  (8 GB laptop's free RAM)   8 pinned   4.305 -> 1.894 s/token  2.27 x (Phase 5 2.13)  acceptance 63 % (1.89/round, 2.87 tokens/round)
                 11 GiB (16 GB laptop's free RAM) 35 pinned   2.819 -> 1.581          1.78 x (Phase 5 1.54)
                 16 GiB                           57 pinned   1.572 -> 1.344          1.17 x (Phase 5 1.02)  (hot: plain 1.36 x Phase 5's cool 1.155)
                 resident                         64 pinned   1.170 -> 1.178          0.99 x (Phase 5 0.79)  (hot: plain 1.41 x Phase 5's cool 0.830;
                                                                                                             capital 1.05, fib 1.28, sentence 0.78 x)
                 (32 greedy tokens x 3 prompts; ids identical at every rung, plain and spec, and equal to Phase 3; peak RSS under every cap with the
                 spec buffers; the plain cost model's worst ratio 1.91 at the hot resident rung, every rung within 2 x; docs/ladder.md refiled with
                 both columns, the Phase 5 file kept as docs/ladder_phase5.md; verify per round 4.64 / 3.80 / 3.16 / 2.75 s against Phase 5's
                 4.96 / 4.39 / 2.69 / 2.50: cheaper where the disk hides the rows, the hot machine on top where it does not)
default        : spec does NOT win at resident (0.99 x on the ladder, 0.97 x on the six-prompt sweep), so the default is left alone: plain decode when
                 nothing is asked, --spec with k = 3 (DEFAULT_SPEC_K). Noted for a later phase: --spec 4 now edges --spec 3 at 8 GiB (2.31 vs
                 2.23 x) because the fourth row is nearly free under the disk, so k could come from the plan on streamed budgets
findings       : 70 an extra activation row is the int8 multiply-accumulate, not the unpack (26 of 50 cycles shareable); 71 the batched
                 kernels leave the ports idle and the sibling thread fills them (12 threads for batches, 6 for the matvec); 72 the tile:
                 16 at hidden width, 8 at intermediate width, 2 always loses, at 4 rows Q4_K gains nothing; 73 prefill 1.5 to 1.8 x the Phase 3
                 path, 3.0 tok/s at 512 tokens, the sibling thread a third of it; 74 identity at every k at resident (30/30), --spec 3 break-even
                 there, the marginal row 0.4 s cool; 75 at 8 GiB the rows cost 0.16 s under the disk, --spec 3 2.23 x, --spec 4 2.31 x, 12/12
                 identical; 76 the ladder 2.27 / 1.78 / 1.17 / 0.99 x, the default stays plain
blocked on     : nothing
did NOT do     : a blocked kernel for the Q8_0 / Q4_0 weights (4.5 % + 1.3 % of the file: ssm_out in 24 layers, a few attention
                 projections; they keep the per-row loop, on 12 threads for batches); the Q8_K row's 64-byte alignment (the probe says a
                 split-line load costs nothing measurable, so it was not changed); an inline-asm inner loop or a 2-rows x 4-activations
                 register tile (the probe puts the floor of any AVX2 arrangement at 41 unshareable cycles per super-block per activation,
                 so neither can reach the gate; not attempted); a look at why the four-accumulator group runs no faster than the per-row
                 loop for Q4_K at four rows (the verify batch; the eight-accumulator group does, so a tile of 4 padded to 8 with duplicated
                 pointers might, at twice the multiplies); a ROW_BLOCK sweep (32 fixed; the tile's L3 re-read it amortises is small at any
                 value above 8); the 6 / 8 / 12 GiB ladder rungs (the brief named 5 / 11 / 16 / resident; the Phase 5 files for the others
                 are kept as *_phase5); a cool-machine rerun (the box was warm throughout: frequency counter 3.1 GHz all-core, membw 25 to
                 29 GB/s; every filed speed-up is a same-minutes A/B); an acceptance sweep with sampling; a VNNI / AVX-512 build (no such
                 CPU here; it is where the remaining 2 x lives, finding 70)
```

## What was built

One kernel, as the brief said: `dot_q8k_t` takes a K-quant weight row and up to sixteen Q8_K activation rows,
unpacks each 256-weight super-block once (codes, pre-shuffled scale broadcasts, mins or scale pairs) and runs the
tile through it with one exact integer accumulator per activation, sub-blocks outer and activations inner in
register groups of eight, four, two and one. Per activation the f32 work is the single-row kernel's in its order,
so every output is the single-row kernel's bit for bit; the scalar reference in `kdot.rs` has the same structure,
and `tests/matmul_t.rs` checks the three paths against each other and against the f64 reference of the Phase 2a
fixtures within the terms budget. `matvec::matmul_t` drives it in row blocks so a weight byte leaves DRAM once
per batch, with the tile capped so its activation rows stay in L2, and `matmul` / `matmul_fn` route every K-quant
batch (the prefill, the verify pass, the MTP re-feed) through it without allocating. Batches of two rows or more
now run on every hardware thread; the single-row matvec keeps the physical cores.

## What was measured, and what it says

The phase asked for 3 x on prefill and 0.2 s per extra verify row. The kernel gives 1.5 to 1.8 x on prefill
(3.0 tokens per second at 512 tokens) and leaves the marginal row at 0.4 to 0.5 s where nothing hides it (0.16 s
under the disk), and the probe in `tools/probe` says this is the core's floor, not the kernel's: a super-block
costs 50 cycles on this i7-9750H, 26 of which a tile can share and 41 of which are the int8 multiply-accumulate
and the f32 tail every activation pays; a `maddubs -> madd -> add` triple costs 2.3 cycles with its load against
1.0 by the port tables, and no reordering of the instruction stream changes that. The sibling hardware thread
fills the idle ports and is worth as much as the blocking itself. What the kernel did change: on the streamed
rungs the rows' compute now hides under the disk almost entirely, so the ladder reads 2.27 x at 5 GiB and 1.78 x
at 11 GiB (2.13 and 1.54 before), and the laptop numbers are 0.53 and 0.63 tokens per second with `--spec 3`;
at resident a round costs 2.35 plain tokens instead of 3.02, which turns speculation from a loss into break-even
and leaves the default as it was. Identity held everywhere it was checked: the blocked prefill's hidden states are
the token-by-token feed's to the bit, the 96 Phase 3 ids are reproduced, every `--spec` run at every `k` at both
rungs emits the plain loop's ids, and the ladder's ids are identical at every rung.

## What comes next

The remaining factor is the instruction set: AVX-512 VNNI folds each triple into one instruction and would take
the unshareable 41 cycles to about 15; on this machine the only lever left in the verify batch is the four-row
group (finding 72), and on streamed budgets `k = 4` is now marginally better than 3. Both are noted for a later
phase; this one stops at the kernel, as instructed.
