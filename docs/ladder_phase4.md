# The ladder (Phase 4)

Qwen3.8-27B Q4_K_M (17.8 GB file) on maneesh-msi, 2026-09-04: the same 3 prompts, 32 greedy tokens each, under memory budgets enforced by a Windows job object (see `docs/data/ladder.txt` for every number and `scripts/ladder.ps1` for the mechanism). s/token, GB/s and %membw are the mean over the 3 prompts; peak RSS is the max over the 3 prompts of the largest of the job's committed-memory peak, the engine's peak working set and the working set sampled every 250 ms. Cost model: t = bytes_ram / 30.06 GB/s + bytes_disk / 3.3 GB/s (unbuffered qd2) + 0.053 s.

| budget | layers pinned | layers streamed | GB/token from disk | s/token | tok/s | GB/s (CPU) | % membw | predicted s/token | measured / predicted | peak RSS (GB) | under budget | compute hidden under reads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6G | 13 | 51 | 12.461 | 3.893 | 0.26 | 4.32 | 14% | 3.974 | 0.98 | 6.049 | yes/yes/yes | 88% |
| 8G | 22 | 42 | 10.298 | 3.318 | 0.30 | 5.07 | 17% | 3.390 | 0.98 | 8.216 | yes/yes/yes | 76% |
| 12G | 40 | 24 | 5.991 | 2.239 | 0.45 | 7.51 | 25% | 2.228 | 1.00 | 12.532 | yes/yes/yes | 53% |
| 16G | 58 | 6 | 1.588 | 1.098 | 0.91 | 15.31 | 51% | 1.041 | 1.06 | 16.943 | yes/yes/yes | 28% |
| 32G | 64 | 0 | 0.000 | 0.747 | 1.34 | 22.51 | 75% | 0.612 | 1.22 | 17.993 | yes/yes/yes | 100% |
| resident | 64 | 0 | 0.000 | 0.741 | 1.35 | 22.68 | 75% | 0.612 | 1.21 | 17.959 | n/a/n/a/n/a | 100% |

Identity gate: ids identical across rungs **yes**; equal to the Phase 3 resident output **yes**. Peak RSS under budget at every rung: **yes**. Worst cost-model ratio 1.22 at 32G.

The 32 ids (identical at every rung):

- `capital`: `11751,13,198,760,6511,314,9564,369,19241,13,198,760,6511,314,14898,369,21047,13,198,760,6511,314,17163,369,23327,13,198,760,6511,314,32208,369`
- `fib`: `198,262,413,307,2564,220,16,25,198,285,460,307,198,262,745,25,198,285,460,73111,1393,12,16,8,478,73111,1393,12,17,8,271,2`
- `sentence`: `733,279,1496,7909,11,91420,2957,25432,279,17865,11,9101,4661,533,33582,82446,321,4172,31717,430,781,87805,13,561,916,1315,314,1439,8177,2218,6067,289`
