# Tiers (Phase 4): the memory plan, pinned arenas, and the streaming ring

`crates/core/src/tier.rs` (plan, arenas, ring), `crates/core/src/os.rs` (unbuffered reads, aligned buffers,
large pages, the job-object cap), `crates/core/src/model.rs` (`Model::load_with`), `crates/cli/src/doctor.rs`,
`scripts/ladder.ps1`. Numbers in `docs/data/doctor_maneesh-msi.txt`, `docs/data/ladder.txt`, `docs/ladder.md`;
findings 54 onward in `docs/payload-vs-doc.md`.

## The plan comes first

`MemoryPlan::compute(&PlanInput, &PlanParams)` is pure arithmetic over the GGUF header (`PlanInput::new`
reads the tensor index, no tensor data) and four parameters: the budget, the positions the KV cache is sized
for, the ring depth, and the drive's sector size. It runs before a single weight byte is allocated, prints a
table (category, bytes, running total, budget, headroom), and `check()` refuses a plan that does not fit
with the shortfall in the message. `aqueduct plan --budget 8G` prints it without loading anything.

Always resident (in this order in the table):

| category | how it is sized | Qwen3.8-27B Q4_K_M |
|---|---|---|
| non-layer arena | the contiguous file range holding `output.weight`, `output_norm.weight`, `token_embd.weight`, aligned | 1,758,130,176 |
| MTP block `blk.64` arena | its span, aligned; loaded but unused this phase | 238,989,312 |
| per-layer small tensors | `RESIDENT_SMALL` (the two norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, `attn_q_norm`, `attn_k_norm`) for all 64 layers, as f32 copies | 10,561,536 |
| DeltaNet carried state | 48 x (48 x 128 x 128 + 10240 x 3) x 4 | 156,893,184 |
| KV cache | 16 layers x 2 x 4 x 256 x 4 = 131,072 bytes per position, times `max_pos` | 536,870,912 at 4096 positions |
| scratch | every buffer `model::State` holds, by the same formulas the constructors use, plus the logits | 2,589,680 at 4096 positions |
| baseline reserve | binary, the parsed GGUF header (the vocabulary arrays), thread stacks, allocator slack: 256 MiB reserved, 41.5 MiB measured after parsing the header | 268,435,456 |

Then the ring, `n_slots` arenas each `arena_bytes(largest streamed layer)`, and tier 1: whole layers from 0
upward. The search runs over the pin count `k` from the top down and takes the largest `k` for which
resident + ring(largest layer of `k..64`) + arenas of `0..k` fits: pinning one more layer can shrink the
ring (the largest layer may move into tier 1), so the feasible set is not a prefix of `k` and a climb from
below can stop early (`tier.rs` unit test). When everything fits there is no ring.

An arena for a `span`-byte range is `align_up(span, sector) + sector` bytes: enough for the aligned superset
of the range wherever it starts modulo the sector (the GGUF aligns tensors to 32 bytes; the sector is
4096). The 4 KiB per arena is in the plan.

The scratch line is a measurement, not an estimate: `State::bytes()` sums the live buffers' capacities and
`tests/tier_plan.rs` asserts it equals the plan formula on the tiny model at two `max_pos` values. The same
test redoes the whole plan by hand from `docs/data/gguf_layout.txt` (spans as literals, a second
implementation of the search) at 6 / 8 / 12 / 16 / 32 GiB and checks that 2 and 3 GiB are refused naming
the shortfall (3 GiB is 290,622,448 bytes short of the 3,511,847,920 the resident set plus a 2-slot ring
needs at 4096 positions).

## One loader for both tiers

Every projection matrix of every layer is a `WeightMat` **view** (`WeightBytes::View { buf: Arc<AlignedBuf>,
off, len }`) into an arena holding the layer's whole contiguous file span (`docs/data/gguf_layout.txt`:
every layer is contiguous, so one read brings the whole layer). A tier-1 layer has its own arena, filled
once at load with one unbuffered read; a tier-2 layer is a view into a ring slot. The small per-layer f32
vectors are resident copies for every layer, whichever tier it is in, so a `DecoderLayer` never changes
after load and the kernels are untouched: `WeightMat::row()` reads through `data()`, which is the owned
vector or the arena window.

`ArenaSource` implements `TensorSource` over one arena: `mat()` returns a view, `vec()` a copy read from the
GGUF the ordinary way. It asserts the split (`is_resident_small`) so the loader and the plan cannot disagree
about which tensors are copies.

The non-layer tensors and the MTP block are arenas too, so the whole load is unbuffered sequential reads:
7.46 GB in 3.0 s (2.49 GB/s) for the 8 GiB plan, against Phase 3's 20 to 30 s buffered load of the same
file.

## The ring

`n_slots` arenas sized for the largest streamed layer. Streamed layer `L` always lands in slot
`(L - first_streamed) % n_slots`: its views are fixed at load, nothing is re-pointed per token. One I/O
thread (`aqueduct-io`, separate from the decode pool, which it does not touch) walks the streamed layers
cyclically: wait for the target slot to be `Free`, one unbuffered read of the layer's aligned superset into
it, mark `Ready(L)`, next. The decode thread waits for `Ready(L)` before running layer `L` and marks the slot
`Free` after it. Both orders are the layer order, so the next occupant of a slot is always `n_slots` layers
on and the read of `L + 1` overlaps the compute of `L`. With two slots the pass boundary can put the last and
the first streamed layer in the same slot (when the streamed count is odd); the refill then starts when the
last layer finishes and overlaps the head and the pinned prefix instead.

The I/O thread allocates nothing in steady state: the overlapped requests, their events and the in-flight
table are preallocated in `DirectFile`, the counters are atomics, the waits are a mutex and a condvar. The
counting-allocator test (`tests/decode_alloc.rs`, part c) sees every thread and still reads zero across 32
decode steps with 7 of the tiny model's 9 layers streaming through 2 slots.

## Rule 4: one read per streamed layer per token

The consumer counts the span bytes it takes (`consumed_bytes`) and `Model::run_layers` / `prefill` assert
after every pass that the delta equals the sum of the streamed spans exactly; `aqueduct run` asserts it
again per token. The I/O thread counts the useful bytes it read (`read_bytes`) and what it asked the OS for
(`io_bytes`, the sector slack: at most 8 KiB per read). `Ring::release` asserts per layer that reads never
exceed consumes by more than the one prefetch in flight. `tests/tier_stream.rs` checks all of it on the tiny
model at 0 / 2 / 5 / 8 pinned layers and 1 / 2 / 3 slots, and that the streamed ids equal the resident ones
bit for bit; the ladder checks it on the real model.

The small resident tensors are also inside the spans, so they are read again with their layer every token
(0.08 % of a layer); the alternative, several reads per layer around them, would cost more in seeks than the
bytes save. The count that rule 4 asserts is the span.

## Unbuffered reads (os.rs)

Windows: `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)`, `ReadFile` with an `OVERLAPPED` per
in-flight request, `GetOverlappedResult(wait)`. The alignment is queried from the file's volume
(`GetFileInformationByHandleEx(FileStorageInfo).PhysicalBytesPerSectorForPerformance`, 4096 here) and
floored at the page. A read is issued as 32 MiB requests with up to `qd` in flight; `doctor` measures qd 1,
2 and 4 with the same function (2.95 / 3.30 / 3.29 GB/s on this drive) and the engine defaults to 2. A read
that runs past the end of the file (the last arena's aligned tail) returns the bytes to EOF, which the
caller checks against the span. Linux: `open(O_DIRECT)` + `pread` chunks (queue depth 1 only; the sysfs
`logical_block_size` of the device, floored at 4096), `mmap` + `madvise(MADV_HUGEPAGE)` for the arenas. The
Linux path compiles (`cargo check --target x86_64-unknown-linux-gnu`) and has not been run.

Arenas are `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)` (page-aligned, zero on demand, so the working set grows
as the reads land). With `MEM_LARGE_PAGES` first when asked: `AdjustTokenPrivileges` tries to enable
`SeLockMemoryPrivilege`, and if the token does not hold it (this user's does not) the fallback to 4 KiB
pages is logged once and the run continues. The difference large pages would make was therefore not
measured; the policy is not changed for the user.

## The cap

`aqueduct run --budget B` sizes the plan. `--job-limit B` makes the engine create a Windows job object
with `JOB_OBJECT_LIMIT_JOB_MEMORY` and `JOB_OBJECT_LIMIT_PROCESS_MEMORY` set to `B` and assign **itself** to
it before anything of size is allocated: from then on a commit past `B` fails and Rust aborts the process.
The engine cannot exceed the cap; the plan is what keeps that from happening. `scripts/ladder.ps1` passes
both flags per rung and records the peak three ways: the job's `PeakJobMemoryUsed` (committed bytes, what
the cap counts), `K32GetProcessMemoryInfo`'s peak working set from inside the engine, and `WorkingSet64`
sampled every 250 ms from the script. Linux (`scripts/ladder.sh`, untested): `systemd-run --scope -p
MemoryMax=B`.

## The cost model

`t = bytes_ram / membw + bytes_disk / diskbw + 0.053`, with `bytes_ram` the weight bytes of the pinned
layers plus the head and one embedding row, `bytes_disk` the streamed spans, membw and diskbw (qd2) from
`doctor`, and 0.053 s the non-matvec constant of `docs/data/nonmatvec_profile.txt`. It treats the streamed
part as disk-bound (its compute is hidden under the next read) and the pinned part as memory-bound. `run
--membw --diskbw` prints the prediction next to the measurement; the ladder files the ratio per rung.
