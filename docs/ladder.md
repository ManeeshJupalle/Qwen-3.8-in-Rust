# The ladder (Phase 5.5: the blocked verify pass; plain and --spec 3 columns)

Qwen3.8-27B Q4_K_M (17.8 GB file) on maneesh-msi, 2026-09-05: the same 3 prompts, 32 greedy tokens each, under memory budgets enforced by a Windows job object (see `docs/data/ladder.txt` for every number and `scripts/ladder.ps1` for the mechanism). The 5 GiB and 11 GiB rungs are the free RAM of an 8 GB and a 16 GB Windows laptop. s/token, GB/s and %membw are the mean over the 3 prompts; peak RSS is the max over the 3 prompts of the largest of the job's committed-memory peak, the engine's peak working set and the working set sampled every 250 ms. Cost model: t = bytes_ram / 30.06 GB/s + bytes_disk / 3.3 GB/s (unbuffered qd2) + 0.053 s. With `--spec 3` the MTP head drafts 3 tokens per round and one batched pass verifies them (`docs/spec.md`); the spec columns are the same prompts and tokens, ids identical.

| budget | note | layers pinned (plain / spec) | layers streamed | GB/token from disk | s/token plain | s/token --spec 3 | speedup | tok/s plain | tok/s spec | accepted per round | acceptance rate | tokens per round | predicted s/token (plain) | measured / predicted | peak RSS plain / spec (GB) | under budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5G | the free RAM of an 8 GB Windows laptop | 9 / 8 | 55 | 13.406 | 4.305 | 1.894 | 2.27 x | 0.23 | 0.53 | 1.89 | 63% | 2.87 | 4.229 | 1.02 | 5.103 / 5.057 | yes/yes/yes / yes/yes/yes |
| 11G | the free RAM of a 16 GB Windows laptop | 36 / 35 | 28 | 7.008 | 2.819 | 1.581 | 1.78 x | 0.35 | 0.63 | 1.89 | 63% | 2.87 | 2.503 | 1.13 | 11.569 / 11.557 | yes/yes/yes / yes/yes/yes |
| 16G |  | 58 / 57 | 6 | 1.588 | 1.572 | 1.344 | 1.17 x | 0.64 | 0.74 | 1.89 | 63% | 2.87 | 1.041 | 1.51 | 16.946 / 16.860 | yes/yes/yes / yes/yes/yes |
| resident |  | 64 / 64 | 0 | 0.000 | 1.170 | 1.178 | 0.99 x | 0.85 | 0.85 | 1.89 | 63% | 2.87 | 0.612 | 1.91 | 17.961 / 18.127 | n/a/n/a/n/a / n/a/n/a/n/a |

Identity gate: ids identical across rungs (plain and --spec) **yes**; equal to the Phase 3 resident output **yes**. Peak RSS under budget at every rung: **yes**. Worst plain cost-model ratio 1.91 at resident.

The 32 ids (identical at every rung):

- `capital`: `11751,13,198,760,6511,314,9564,369,19241,13,198,760,6511,314,14898,369,21047,13,198,760,6511,314,17163,369,23327,13,198,760,6511,314,32208,369`
- `fib`: `198,262,413,307,2564,220,16,25,198,285,460,307,198,262,745,25,198,285,460,73111,1393,12,16,8,478,73111,1393,12,17,8,271,2`
- `sentence`: `733,279,1496,7909,11,91420,2957,25432,279,17865,11,9101,4661,533,33582,82446,321,4172,31717,430,781,87805,13,561,916,1315,314,1439,8177,2218,6067,289`

## Thermal state of this run (Phase 5.5)

The four rungs ran back to back in one pass (`scripts/ladder.ps1 -Spec 3 -Budgets 5G,11G,16G,resident`, 22:51 to 23:22,
membw 30.6 GB/s before and 29.7 after). The disk-bound rungs keep the machine cool: the 5 and 11 GiB plain tokens,
4.305 and 2.819 s, are within 1 % of Phase 5's 4.316 and 2.790. The RAM-bound rungs heat it: the 16 GiB and resident
plain tokens, 1.572 and 1.170 s, are 1.36 and 1.41 x Phase 5's cool 1.155 and 0.830 (finding 40; the same machine
decoded at 0.926 s per token in this session's end-to-end run and 0.75 to 0.91 on the first prompts of the resident
acceptance sweep). Every speed-up pairs a plain and a `--spec` run of the same prompt taken minutes apart, so the
ratios hold in either state; the absolute 16 GiB and resident numbers are not comparable with Phase 5's, and
`docs/ladder_phase5.md` keeps that file.

## Against Phase 5 (the same rungs, the un-blocked verify pass)

| rung | speed-up Phase 5 | speed-up Phase 5.5 | verify per round Phase 5 | Phase 5.5 | plain token Phase 5 | Phase 5.5 |
|---|---|---|---|---|---|---|
| 5 GiB | 2.13 x | 2.27 x | 4.957 | 4.644 | 4.316 | 4.305 |
| 11 GiB | 1.54 x | 1.78 x | 4.393 | 3.800 | 2.790 | 2.819 |
| 16 GiB | 1.02 x | 1.17 x | 2.692 | 3.159 (hot) | 1.155 | 1.572 (hot) |
| resident | 0.79 x | 0.99 x | 2.504 | 2.752 (hot) | 0.830 | 1.170 (hot) |

Where the disk is the bottleneck the round got 0.3 to 0.6 s cheaper (the three extra rows' compute now hides
under the read almost entirely, finding 75); where it is not, the round costs 2.35 plain tokens instead of 3.02,
which turns the resident rung from a loss into break-even (0.99 x: `capital` 1.05, `fib` 1.28, `sentence` 0.78).
Speculation does not win where the model fits, so plain decode stays the default and `--spec` keeps `k = 3`
(finding 76).

## The speculative round against the cost model (Phase 5.5)

`t_round = bytes_ram / membw_eff + bytes_disk / diskbw + c_mtp x k + c_verify x (k + 1)` with `c_mtp = 0.085 s` and
`c_verify = 0.527 s` (fitted at the hot resident rung: `(2.752 - 1.170) / 3`; Phase 5's cool fit was 0.558, and the
gate of this phase, 0.2 s per extra row, is not met); `tools/spec_cost_model.py --rungs 5G,11G,16G,resident`,
`docs/data/spec_cost_model.txt`. The sum overstates the streamed rungs by 20 to 25 % because the rows' compute
overlaps the disk read there.

| rung | pinned (spec) | plain s/token | spec s/token | speedup | s/round measured | verify | MTP re-feed | chain/step | replay | tokens/round | predicted s/round | measured / predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5G | 9 (8) | 4.305 | 1.894 | 2.27 x | 5.034 | 4.644 | 0.118 | 0.085 | 0.077 | 2.74 | 6.669 | 0.75 |
| 11G | 36 (35) | 2.819 | 1.581 | 1.78 x | 4.201 | 3.800 | 0.122 | 0.085 | 0.083 | 2.74 | 5.182 | 0.81 |
| 16G | 58 (57) | 1.572 | 1.344 | 1.17 x | 3.572 | 3.159 | 0.125 | 0.087 | 0.090 | 2.74 | 3.935 | 0.91 |
| resident | 64 (64) | 1.170 | 1.178 | 0.99 x | 3.131 | 2.752 | 0.116 | 0.083 | 0.075 | 2.74 | 3.534 | 0.89 |
