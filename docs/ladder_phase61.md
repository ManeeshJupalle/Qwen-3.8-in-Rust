# The ladder, re-run in Phase 6.1 (hot, right after the llama.cpp sweep)

Qwen3.8-27B Q4_K_M (17.8 GB file) on maneesh-msi, 2026-09-06: the same 3 prompts, 32 greedy tokens each, under memory budgets enforced by a Windows job object (see `docs/data/ladder_phase61.txt` for every number and `scripts/ladder.ps1` for the mechanism; this is the same-session rerun paired with `docs/data/vs_llamacpp.txt`, taken with the laptop hot after two hours of llama.cpp runs: the cool numbers are in `docs/ladder.md`). The 5 GiB and 11 GiB rungs are the free RAM of an 8 GB and a 16 GB Windows laptop. s/token, GB/s and %membw are the mean over the 3 prompts; peak RSS is the max over the 3 prompts of the largest of the job's committed-memory peak, the engine's peak working set and the working set sampled every 250 ms. Cost model: t = bytes_ram / 30.06 GB/s + bytes_disk / 3.3 GB/s (unbuffered qd2) + 0.053 s. With `--spec 3` the MTP head drafts 3 tokens per round and one batched pass verifies them (`docs/spec.md`); the spec columns are the same prompts and tokens, ids identical.

| budget | note | layers pinned (plain / spec) | layers streamed | GB/token from disk | s/token plain | s/token --spec 3 | speedup | tok/s plain | tok/s spec | accepted per round | acceptance rate | tokens per round | predicted s/token (plain) | measured / predicted | peak RSS plain / spec (GB) | under budget |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5G | the free RAM of an 8 GB Windows laptop | 9 / 8 | 55 | 13.406 | 4.551 | 2.336 | 1.95 x | 0.22 | 0.43 | 1.89 | 63% | 2.87 | 4.229 | 1.08 | 5.104 / 5.057 | yes/yes/yes / yes/yes/yes |
| 8G |  | 22 / 21 | 42 | 10.298 | 4.020 | 2.491 | 1.61 x | 0.25 | 0.40 | 1.89 | 63% | 2.87 | 3.390 | 1.19 | 8.219 / 8.172 | yes/yes/yes / yes/yes/yes |
| 11G | the free RAM of a 16 GB Windows laptop | 36 / 35 | 28 | 7.008 | 3.379 | 2.384 | 1.42 x | 0.30 | 0.42 | 1.89 | 63% | 2.87 | 2.503 | 1.35 | 11.568 / 11.557 | yes/yes/yes / yes/yes/yes |
| resident |  | 64 / 64 | 0 | 0.000 | 1.534 | 1.446 | 1.06 x | 0.65 | 0.69 | 1.89 | 63% | 2.87 | 0.612 | 2.51 | 17.962 / 18.128 | n/a/n/a/n/a / n/a/n/a/n/a |

Identity gate: ids identical across rungs (plain and --spec) **yes**; equal to the Phase 3 resident output **yes**. Peak RSS under budget at every rung: **yes**. Worst plain cost-model ratio 2.51 at resident.

The 32 ids (identical at every rung):

- `capital`: `11751,13,198,760,6511,314,9564,369,19241,13,198,760,6511,314,14898,369,21047,13,198,760,6511,314,17163,369,23327,13,198,760,6511,314,32208,369`
- `fib`: `198,262,413,307,2564,220,16,25,198,285,460,307,198,262,745,25,198,285,460,73111,1393,12,16,8,478,73111,1393,12,17,8,271,2`
- `sentence`: `733,279,1496,7909,11,91420,2957,25432,279,17865,11,9101,4661,533,33582,82446,321,4172,31717,430,781,87805,13,561,916,1315,314,1439,8177,2218,6067,289`
