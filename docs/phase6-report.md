# Phase 6 report (ship v0.1.0)

Commits 3c34eb7 (the phase), 76554b6 and 6cfe71f (the clean-clone fixes and transcripts), and f64f95a (the
Linux lint fix, the commit `v0.1.0` tags), on `main` of the repository renamed to `Qwen3.8-in-c` this session
(the crate and the binary stay `aqueduct`). The history was rewritten twice this session at the user's
request, to remove attribution trailers from earlier commits and then a scratchpad path from the data
files' older versions, so every hash before this paragraph differs from what CI ran on (the trees are the
same code; CI runs 34019968088 and 34020854109 ran on the pre-rewrite hashes 99693d7 and c4d045e, and the
release run 34021676564 on the tag's earlier object). No engine feature, no kernel, no GPU. The doctor's outputs, the gate audit, the
clean-clone and demo transcripts are under `docs/data/` (`doctor_phase6_*.txt`, `avx2_gate_audit.txt`,
`clean_clone_maneesh-msi.txt`, `demo_maneesh-msi.txt`); the README is the deliverable.

```
PHASE 6 REPORT
crate name     : aqueduct, crates.io free (and aqueduct-core free) -> chosen aqueduct; the GitHub repository was renamed
                 Qwen3.8-in-c by the user mid-phase (remote, crate metadata, README and CI follow; the binary is aqueduct)
license        : MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE, `license` in both crates); model license linked y
                 (Apache-2.0, huggingface.co/Qwen/Qwen3.8-27B); the GGUF credited to bartowski with its repo link
doctor         : verdict on this box (no model, NVMe volume, 17.6 GiB free) "Good: everything fits in RAM; expect ~0.6 s/token
                 (--spec 3 would not help here: ~1.2) (the drive only affects the load: about 5 s for 17.8 GB at 3.38 GB/s)";
                 at --budget 11G on the NVMe "Good: expect ~1.9 s/token with --spec 3 (~2.6 plain)" (33 of 64 pinned at 4096
                 positions); at --budget 11G on the D: spinning disk (SATA ST1000LM049, seek penalty yes, 0.11 GB/s unbuffered)
                 "Not recommended: spinning HDD, expect ~27.0 s/token with --spec 3 (~69.4 plain)"; next command printed with the
                 download (curl.exe, --create-dirs), the sha256 (certutil / --sha256), the three Qwen files and the chat line;
                 (model present) the same "Good ... ~0.6 s/token" with the size checked against the supported file and the
                 sha256 verified once with --sha256 (e103abf9...52b, 160 s); every run filed (docs/data/doctor_phase6_*.txt)
readme         : sections 9/9 in the brief's order (pitch, ladder, quickstart Windows/Linux/source, how it works, cost model,
                 limitations, not yet, validation, credits); numbers backed: every performance number carries a link to its
                 docs/data file (24 distinct files linked, all present); limitations items 21/13+
ci             : windows green (17.4 min), ubuntu green (14.5 min) on run 34020854109 (the lint-fix commit, f64f95a after the rewrite); runs 1-3
                 failed only at ubuntu's clippy on two pre-existing lines of os.rs while their cargo test on ubuntu passed (the Linux
                 path's first execution, tiny-model O_DIRECT streaming included); binaries 2 (release run 34021676564 on the v0.1.0 tag:
                 aqueduct-v0.1.0-x86_64-pc-windows-msvc.exe 6,539,264 bytes, aqueduct-v0.1.0-x86_64-unknown-linux-gnu 9,310,440 bytes,
                 x86-64-v3, built in 3.7 and 2.1 min), sha256 filed y (a86c216b...5ece and c3f40d92...bb87: .sha256 assets, the release
                 notes, docs/data/release_v0.1.0_sha256.txt; both re-verified after download, and the CI-built Windows binary runs here:
                 --version, doctor at 11G "Good: expect ~1.8 s/token with --spec 3"); publish --dry-run ok (both crates verified)
clean clone    : build 91 s (fresh clone, cargo build --release); doctor ok (told the user the 3 Qwen files were missing and printed
                 the commands); chat at 8G --spec 3: 0.55 tok/s (19-token prompt 8.6 s, 7 tokens in 12.8 s, 2.00 drafts accepted per
                 round); README fixes needed: 2 -- `curl` is Invoke-WebRequest in Windows PowerShell 5.1 (now curl.exe);
                 `target/release/aqueduct` does not run under cmd.exe (now target\release\aqueduct for Windows)
failure modes  : wrong path one-line ("error: load C:\models\does-not-exist.gguf: io error: The system cannot find the file
                 specified. (os error 2)", exit 1, run and chat); 3G budget one-line in chat ("error: load ...: memory plan does not
                 fit: the always-resident set (2972470256 bytes) plus a 2-slot ring (539377664 bytes) needs 3511847920 bytes =
                 3.27 GiB, the budget is 3221225472 bytes = 3.00 GiB: short by 290622448 bytes = 0.27 GiB", exit 1); `run --max-tokens 2`
                 at 3G is NOT refused: it sizes the KV cache to 5 positions, fits at 0 pinned / 64 streamed and decodes at 4.8 s/token
                 (README limitation 13 says so)
demo           : scripts/demo.ps1 runs, duration 131 s (doctor ~25 s, plan, a 30-token prompt in 12.6 s, 65 tokens at 0.61 tok/s at
                 11G --spec 3, 41 % acceptance on prose)
engine changes : two lint-only lines in crates/core/src/os.rs (Linux branch, found by CI's ubuntu clippy): `c.as_ptr() as *const i8`
                 -> `.cast()`, and `qd.max(1).min(1)` -> the constant 1 it already was, with `let _ = qd;`. No behaviour change.
                 Outside the engine: crates/cli (main.rs: the cpuid gate before any code, --version; doctor.rs rewritten; disk.rs
                 and known.rs new; run.rs / chat.rs: the load error names the path, a Linux default model path), crates/core/tests
                 (8 files: 11 tests skip with a message when the 17.8 GB file is absent, AQUEDUCT_REQUIRE_MODEL=1 makes that a
                 failure), Cargo metadata, scripts/ladder.ps1's hint, ARCHITECTURE.md's session line
did NOT do     : run on a machine that is not the author's (the clean clone ran on the same laptop); publish to crates.io (dry-run only);
                 a static musl Linux binary; re-measure the ladder with the x86-64-v3 build; a tool-call template fixture; embed the
                 tokenizer, template and generation config (three curl commands instead); the Windows release binary's own sha256
                 verified on a second machine; a Linux doctor run (the disk-bus sysfs path is compiled and lint-clean, unrun)
```

## What the doctor is now

`aqueduct doctor` runs without the model: it plans from a built-in table of the supported file's layout
(`crates/cli/src/known.rs`, generated from the Phase 0 fixtures and checked against them and against the real
file whenever it is present), measures the machine (physical cores, AVX2 and F16C, RAM and free RAM, memory
bandwidth), asks the OS what drive the model path is on (`IOCTL_STORAGE_QUERY_PROPERTY` on Windows: the bus
and the seek penalty; sysfs on Linux) and measures unbuffered reads on it (the model's own layers when the
file is there, a 1 GiB scratch file on the same volume when not; on the C: root, which refuses the scratch
file, it falls back to the temp directory on the same volume). Then the plan for the free RAM rounded down
to half a GiB, the prediction table (plain and `--spec 3` at 6 / 8 / 12 / 16 GiB, the recommended budget and
resident), one verdict line and the next commands with every path filled in. The four verdict sentences of
the brief are all reachable and three were exercised on this machine (`Good`, `Not recommended`, `Cannot run
at this free RAM` through the plan's refusal; `Cannot run: AVX2 missing` is the one this CPU cannot show, so
the gate in `main` was audited instead).

## The AVX2 gate

The release binaries are compiled for x86-64-v3, so the check must run before any code that the compiler
may have vectorised. `main` calls `require_avx2` first (cpuid leaves 1 and 7, XGETBV for the YMM state, a raw
`write_all` to stderr, `exit(2)`) and only then `real_main`. `tools/asm_audit.py` walks the emitted assembly
from the C entry and from std's `lang_start` closure (which reaches `aqueduct::main` through a register call)
to the `cpuid` and lists every VEX or BMI mnemonic on the way: none (`docs/data/avx2_gate_audit.txt`, seven
symbols walked; the gate functions' whole bodies are clean too). The one thing not done is running it on a
CPU without AVX2; there is none here.

## CI

`.github/workflows/ci.yml` runs `cargo test --workspace` and `clippy -D warnings` on `windows-latest` and
`ubuntu-latest` without the 17.8 GB model. The tests need three small files from Qwen's Hugging Face repo and
the 140 MB tiny oracle GGUF, which the converter built in Phase 2a and cannot be regenerated on a runner;
it is a release asset (`fixtures-v1`, sha256-checked by the workflow). Eleven tests open the real file and
now skip with a printed line when it is absent. The ubuntu job's `cargo test` is the first time the Linux
`O_DIRECT` path, the sysfs sector query and the streaming ring have executed anywhere: green on the first
run. Its clippy found two lines of the Linux branch of `os.rs` that had never been linted (the engine change
above). `release.yml` builds both binaries on a `v*` tag, writes a `.sha256` next to each and publishes the
release with `.github/release-notes.md` (the ladder) plus the checksums.

## What the clean clone taught

Two README bugs, both Windows shell facts: in Windows PowerShell 5.1 `curl` is an alias of `Invoke-WebRequest`,
which rejects `-L --create-dirs`, so the commands (and the doctor's printed ones) say `curl.exe`; and
`cmd.exe` does not run `target/release/aqueduct` (it reads the slashes as switches), so the Windows line uses
backslashes. Everything else worked from the README alone: the clone built in 91 s, the doctor told the
user exactly which of the three Qwen files were missing and printed the commands, the downloads landed where
the defaults look, and the chat at 8 GiB answered and stopped on its own.
