# The ladder (Phase 4, with the Phase 5 --spec column)

Qwen3.8-27B Q4_K_M (17.8 GB file) on maneesh-msi, 2026-09-05: the same 3 prompts, 32 greedy tokens each, under memory budgets enforced by a Windows job object (see `docs/data/ladder.txt` for every number and `scripts/ladder.ps1` for the mechanism). The 5 GiB and 11 GiB rungs are the free RAM of an 8 GB and a 16 GB Windows laptop. s/token, GB/s and %membw are the mean over the 3 prompts; peak RSS is the max over the 3 prompts of the largest of the job's committed-memory peak, the engine's peak working set and the working set sampled every 250 ms. Cost model: t = bytes_ram / 30.06 GB/s + bytes_disk / 3.3 GB/s (unbuffered qd2) + 0.053 s. With `--spec 3` the MTP head drafts 3 tokens per round and one batched pass verifies them (`docs/spec.md`); the spec columns are the same prompts and tokens, ids identical.

| budget | note | layers pinned (plain / spec) | layers streamed | GB/token from disk | s/token plain | s/token --spec 3 | speedup | tok/s plain | tok/s spec | accepted per round | acceptance rate | tokens per round | predicted s/token (plain) | measured / predicted | peak RSS plain / spec (GB) | under budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5G | the free RAM of an 8 GB Windows laptop | 9 / 8 | 55 | 13.406 | 4.316 | 2.028 | 2.13 x | 0.23 | 0.49 | 1.89 | 63% | 2.87 | 4.229 | 1.02 | 5.104 / 5.057 | yes/yes/yes / yes/yes/yes |
| 6G |  | 13 / 12 | 51 | 12.461 | 4.095 | 1.979 | 2.07 x | 0.24 | 0.51 | 1.89 | 63% | 2.87 | 3.974 | 1.03 | 6.051 / 6.005 | yes/yes/yes / yes/yes/yes |
| 8G |  | 22 / 21 | 42 | 10.298 | 3.628 | 1.812 | 2.00 x | 0.28 | 0.55 | 1.89 | 63% | 2.87 | 3.390 | 1.07 | 8.218 / 8.172 | yes/yes/yes / yes/yes/yes |
| 11G | the free RAM of a 16 GB Windows laptop | 36 / 35 | 28 | 7.008 | 2.790 | 1.814 | 1.54 x | 0.36 | 0.55 | 1.89 | 63% | 2.87 | 2.503 | 1.12 | 11.568 / 11.557 | yes/yes/yes / yes/yes/yes |
| 12G |  | 40 / 39 | 24 | 5.991 | 2.690 | 2.050 | 1.31 x | 0.37 | 0.49 | 1.89 | 63% | 2.87 | 2.228 | 1.21 | 12.535 / 12.504 | yes/yes/yes / yes/yes/yes |
| 16G |  | 58 / 57 | 6 | 1.588 | 1.155 | 1.129 | 1.02 x | 0.87 | 0.89 | 1.89 | 63% | 2.87 | 1.041 | 1.11 | 16.946 / 16.860 | yes/yes/yes / yes/yes/yes |
| resident |  | 64 / 64 | 0 | 0.000 | 0.830 | 1.057 | 0.79 x | 1.21 | 0.95 | 1.89 | 63% | 2.87 | 0.612 | 1.36 | 17.961 / 18.128 | n/a/n/a/n/a / n/a/n/a/n/a |

Identity gate: ids identical across rungs (plain and --spec) **yes**; equal to the Phase 3 resident output **yes**. Peak RSS under budget at every rung: **yes**. Worst plain cost-model ratio 1.36 at resident.

The 32 ids (identical at every rung):

- `capital`: `11751,13,198,760,6511,314,9564,369,19241,13,198,760,6511,314,14898,369,21047,13,198,760,6511,314,17163,369,23327,13,198,760,6511,314,32208,369`
- `fib`: `198,262,413,307,2564,220,16,25,198,285,460,307,198,262,745,25,198,285,460,73111,1393,12,16,8,478,73111,1393,12,17,8,271,2`
- `sentence`: `733,279,1496,7909,11,91420,2957,25432,279,17865,11,9101,4661,533,33582,82446,321,4172,31717,430,781,87805,13,561,916,1315,314,1439,8177,2218,6067,289`

## Thermal state of this run

The 5 to 12 GiB rungs were measured in a mildly throttled state (membw 27 to 28 GB/s at 6 threads; finding 40), the 16 GiB and resident rungs after the machine had cooled (28.7 GB/s before them, 29.95 after): their plain tokens, 1.155 and 0.830 s, are within 6 and 12 % of Phase 4's cool 1.098 and 0.741. Every speedup pairs a plain and a `--spec` run taken minutes apart, so the ratios hold in either state; the absolute s/token of the streamed rungs are a few percent slower than Phase 4's. `docs/ladder_phase4.md` and `docs/data/ladder_phase4.txt` keep the Phase 4 files.

## The speculative round against the cost model (Phase 5.4)

`t_round = bytes_ram / membw_eff + bytes_disk / diskbw + c_mtp x k + c_verify x (k + 1)` with `c_mtp = 0.083 s` (the mean chained draft step over the rungs: the MTP block and the shared head, both resident) and `c_verify = 0.558 s` (fitted at resident as the marginal compute of one extra verify row, where nothing hides it); `membw_eff` is each rung's own plain RAM rate so the first two terms reproduce the plain token. Filed by `tools/spec_cost_model.py` in `docs/data/spec_cost_model.txt`. The sum overstates the streamed rungs by about 20 % because the extra rows' compute overlaps the disk read there; the 12 GiB row is the hot-state pass (see above).

| rung | pinned (spec) | plain s/token | spec s/token | speedup | s/round measured | verify | MTP re-feed | chain/step | replay | tokens/round | predicted s/round | measured / predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5G | 9 (8) | 4.316 | 2.028 | 2.13 x | 5.389 | 4.957 | 0.152 | 0.089 | 0.078 | 2.74 | 6.798 | 0.79 |
| 6G | 13 (12) | 4.095 | 1.979 | 2.07 x | 5.258 | 4.829 | 0.154 | 0.088 | 0.074 | 2.74 | 6.577 | 0.80 |
| 8G | 22 (21) | 3.628 | 1.812 | 2.00 x | 4.816 | 4.437 | 0.138 | 0.079 | 0.059 | 2.74 | 6.109 | 0.79 |
| 11G | 36 (35) | 2.790 | 1.814 | 1.54 x | 4.820 | 4.393 | 0.152 | 0.091 | 0.066 | 2.74 | 5.272 | 0.91 |
| 12G | 40 (39) | 2.690 | 2.050 | 1.31 x | 5.448 | 4.926 | 0.186 | 0.108 | 0.081 | 2.74 | 5.171 | 1.05 |
| 16G | 58 (57) | 1.155 | 1.129 | 1.02 x | 2.999 | 2.692 | 0.106 | 0.064 | 0.053 | 2.74 | 3.636 | 0.82 |
| resident | 64 (64) | 0.830 | 1.057 | 0.79 x | 2.807 | 2.504 | 0.108 | 0.064 | 0.047 | 2.74 | 3.311 | 0.85 |
