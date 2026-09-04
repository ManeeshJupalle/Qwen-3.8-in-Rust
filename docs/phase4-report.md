# Phase 4 report (tiers and streaming)

Commits c435d7f .. (this one) on `main`, 2026-09-04. Design in `docs/tiers.md`; numbers in `docs/data/`
(`doctor_maneesh-msi.txt`, `ladder.txt`) and `docs/ladder.md`; findings 54 to 61 in `docs/payload-vs-doc.md`;
the runner is `scripts/ladder.ps1` (`scripts/ladder.sh` for Linux, untested). Machine: i7-9750H laptop
(6 cores / 12 threads), 32 GB DDR4-2667, the GGUF on the C: NVMe SSD (Crucial P310), Windows 11. No MTP, no
sampling, no chat template, no GPU: greedy decode as in Phase 3.

```
PHASE 4 REPORT
tests          : 65/65 non-ignored (18 unit + 47 integration over 17 test targets; the new ones: tier_plan 2, tier_stream 2,
                 decode_alloc extended with a streamed part c, os and tier unit tests) + 2 ignored model-scale gates unchanged;
                 clippy --all-targets clean; the core crate also compiles for x86_64-unknown-linux-gnu (never run there)
plan           : budgets 6/8/12/16/32 GiB -> pinned layers 11/20/38/56/64 and streamed 53/44/26/8/0 at the `plan` default of 4096
                 positions (13/22/40/58/64 and 51/42/24/6/0 at the ladder's 38 to 72 positions), ring 2 slots x 257.2 MiB
                 (269,688,832 bytes each: the 269,681,536-byte layers aligned to the 4096-byte sector); resident set 2.97 GB
                 (non-layer arena 1.758 GB, blk.64 0.239 GB, small per-layer tensors 10.6 MB, DeltaNet state 156.9 MB, KV cache
                 131,072 bytes per position, scratch 2.59 MB, baseline reserve 256 MiB); the table adds up to the byte against a
                 second implementation of the arithmetic in tests/tier_plan.rs; refusal at 3 GiB: "memory plan does not fit: the
                 always-resident set (2972470256 bytes) plus a 2-slot ring (539377664 bytes) needs 3511847920 bytes = 3.27 GiB,
                 the budget is 3221225472 bytes = 3.00 GiB: short by 290622448 bytes = 0.27 GiB" (2 GiB: short by 1.27 GiB);
                 `aqueduct plan --budget X` prints it in 160 ms having read 0 tensor bytes
disk           : C: NVMe, unbuffered (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, 32 MiB requests) sequential reads of one
                 layer: 2.95 GB/s qd1 (runs 1.72 2.95 2.95 2.84 2.93), 3.30 GB/s qd2 (3.29 3.30 3.30 3.30 3.30), 3.29 qd4,
                 3.28 "cold" after 2 GiB of other data; largest layer 257.19 MiB (269,681,536 bytes, layers 0, 1, 2, 4, 5, 6,
                 10, 22, 34, 46, 56, 57, 58, 60, 61, 62); sector 4096 bytes (FILE_STORAGE_INFO PhysicalBytesPerSectorForPerformance,
                 queried; logical 512); large pages: fallback, "SeLockMemoryPrivilege is not held by this user", logged once and
                 the run continues on 4 KiB pages; the policy was not changed, so the difference was NOT measured
streaming      : bytes/token measured 12.461 / 10.298 / 5.991 / 1.588 GB at 6 / 8 / 12 / 16 GiB vs expected the same to the byte
                 (12,460,584,192 / 10,297,762,944 / 5,990,919,936 / 1,587,863,040 = the streamed spans; asserted after every
                 pass inside the model and again per token by `run`; per layer, reads never exceed consumes by more than the one
                 prefetch in flight); the ring reads at 3.25 to 3.31 GB/s, what doctor measured; prefetch overlap 21 / 22 / 24 /
                 46 % of the read time hidden, which in this disk-bound regime is the compute share, and 88 / 76 / 53 / 28 % of
                 the compute hidden under reads; the drive is busy 97 / 94 / 82 / 45 % of the wall time; 3 slots at a fixed split:
                 3.348 vs 3.31-3.33 s/token at the 8 GiB split (no gain), 1.075 vs 1.086-1.116 at the 16 GiB split (2 %); zero
                 allocations across 32 decode steps with 7 of 9 tiny-model layers streaming, the I/O thread included
ladder         : budget   pinned streamed GB/token s/token  GB/s(CPU) %membw peak RSS (max of job commit / engine WS / sampled)
                 6 GiB    13     51       12.461   3.893    4.32      14 %   6.049 GB  (cap 6.442: under)
                 8 GiB    22     42       10.298   3.318    5.07      17 %   8.216 GB  (cap 8.590: under)
                 12 GiB   40     24        5.991   2.239    7.51      25 %  12.532 GB  (cap 12.885: under)
                 16 GiB   58      6        1.588   1.098   15.31      51 %  16.943 GB  (cap 17.180: under)
                 32 GiB   64      0        0       0.747   22.51      75 %  17.993 GB  (cap 34.360: under)
                 resident 64      0        0       0.741   22.68      75 %  17.959 GB  (no cap)
                 (mean over the 3 prompts; membw 30.06 GB/s; loads 2.2 / 3.0 / 4.9 / 6.6 / 7.4 / 7.3 s, unbuffered at 2.5 to 2.9 GB/s)
identity       : ids identical across rungs yes (all 3 prompts, all 6 rungs); vs Phase 3 resident: match (all 96 ids equal
                 tests/fixtures/ladder_expected_ids.json, taken from docs/data/phase3_timing.txt)
cost model     : t = bytes_ram / 30.06 + bytes_disk / 3.30 + 0.053: predicted 3.974 / 3.390 / 2.228 / 1.041 / 0.612 / 0.612 vs
                 measured 3.893 / 3.318 / 2.239 / 1.098 / 0.747 / 0.741, ratios 0.98 / 0.98 / 1.00 / 1.06 / 1.22 / 1.21;
                 worst 1.22 at 32 GiB: the RAM term assumes the kernels stream at the full membw and they reach 74 to 77 %
                 (findings 49, 52); every rung inside the 2 x gate, the streamed ones inside 6 %
findings       : 54 the ladder (3.9 to 0.74 s/token, same ids everywhere, RSS under every cap); 55 cost model within 6 % where
                 the disk dominates, 1.2 x optimistic where RAM does; 56 the drive is busy 94 to 97 % of the wall under
                 streaming, the engine runs at the drive's sequential rate; 57 two slots suffice where the disk is the
                 bottleneck, a third gives 2 % at the 16 GiB split and costs a pinned layer at a fixed budget; 58 unbuffered
                 arena loads at 2.5 to 2.9 GB/s, 7 s for the resident model vs Phase 3's 20 to 30 s; 59 the 256 MiB baseline
                 reserve is generous by ~0.2 GB (measured overhead 35 to 60 MB), kept; 60 the pin-count search must run from
                 the top down (pinning a layer can shrink the ring), a unit test caught the climb-from-below version; 61 two
                 PowerShell facts (null ExitCode without touching the handle; case-insensitive parameter/variable clash coerced
                 the parsed fixture to a string) broke the first two ladder runs and were fixed in the script, which can now
                 re-render both files from the run's saved stats (-FromStats: how the filed ladder was produced)
blocked on     : nothing
did NOT do     : measure large pages (the privilege is not held and the policy stays); run the Linux path (O_DIRECT, mmap +
                 MADV_HUGEPAGE, scripts/ladder.sh: compiled, untested, queue depth 1 only); enforce the cap from PowerShell with
                 CreateProcess-suspended (the engine puts itself into the job object before allocating, --job-limit; the script
                 states it); a sequence-based slot mapping that would remove the pass-boundary slot collision (fixed mapping
                 kept: the odd-count rung, 6 GiB, shows no penalty at 0.98 of the model); refine the cost model's RAM term to
                 the kernels' 75 % (the formula was specified; the ratio is filed); shrink the baseline reserve; use the MTP
                 block (loaded, unused); the sampled-WorkingSet64 column of the filed ladder (the run predates the sidecar
                 file the script now writes; the column is bounded by the engine's own peak working set and shows 0.00)
```

## What the next session should do first

1. The ladder says the streamed rungs run at the drive's rate and the resident one at 75 % of the bus, so
   the pitch's "16 GB laptop" number is 1.10 s per token at 58 pinned layers with 6 streamed (0.91 tok/s),
   and 0.74 s fully resident. A SATA drive at 0.5 GB/s would make the 6 streamed layers 3.2 s alone: `doctor`
   should say so before the download (Phase 6), and the report format should carry the drive class.
2. The 3.7 GB that the always-resident set costs (non-layer 1.76 GB, of which the 1.04 GB output head and the
   0.72 GB embedding table are the bulk; blk.64 0.24 GB; KV at 4096 positions 0.54 GB) is worth revisiting:
   the embedding table is read one row per token and could be streamed (or memory-mapped) for 0.7 GB, and
   blk.64 is unused until Phase 5. Either is 2 to 3 more pinned layers at every rung.
3. Phase 5 (MTP, sampling, chat template) can start on this loader: `Model::load_with` with a budget is the
   only entry point, and every layer is a view whichever tier it is in.
