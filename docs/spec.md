# Speculative decoding (Phase 5.2): the round, verification, rollback, and what it costs

`crates/core/src/spec.rs` implements this; `docs/mtp.md` is the draft head it drives; `crates/core/src/sample.rs`
the sampler it verifies against. Rules (from the phase brief): greedy with `--spec` emits byte-identical ids to
greedy without it; the decode step allocates nothing; each streamed weight byte comes from disk at most once per
verification round; the memory plan sizes everything before allocation.

## Positions and names

The main model has **consumed** positions `0..P-1` (their tokens are in the KV caches and the DeltaNet states)
and has **emitted** `x_P` from the logits of position `P-1`; `x_P` is not yet consumed. `h_j` is the target's
post-final-norm hidden at position `j` (what the lm_head read to emit `x_{j+1}`, `docs/mtp.md`). Round `r`:

1. **Draft.** The MTP produces `d_1..d_k`, its guesses for `x_{P+1}..x_{P+k}` (how, and what its cache holds, in
   `docs/mtp.md`).
2. **Verify.** One batched forward of the `k + 1` tokens `[x_P, d_1, ..., d_k]` at positions `P..P+k` through the
   64 layers (`Model::forward_batch`, the allocation-free form of the Phase 3.3 batched prefill: every projection a
   `matmul` over the `k + 1` activations, so every weight row is read once for the round, streamed layers
   included), then the lm_head on all `k + 1` rows. Row `i` of the logits is the model's prediction for position
   `P + i + 1` given the batch prefix. Bit for bit, row `i` equals what `forward_token` would have produced after
   consuming `x_P, d_1, ..., d_i` one at a time (finding 41: same `one_row` kernel per row, same per-token code
   for the conv, the recurrence and the attention).
3. **Accept.** Greedy: `m` = the length of the longest prefix with `argmax(logits_i) == d_{i+1}`, `i = 0..k-1`.
   The bonus token is `argmax(logits_m)`: the model's own next token after the accepted prefix (for `m = k` the
   token after all drafts). The round emits `d_1..d_m` and the bonus, `m + 1` tokens, all of them exactly the
   tokens greedy decoding would have emitted at those positions. Sampling: the rejection-sampling rule below,
   same shape (`m` accepted, one resampled or sampled token).
4. **Roll back.** The batch consumed positions `P..P+k`; only `P..P+m` stay consumed (`m + 1` of `k + 1`).
   KV caches (16 layers): `len = P + m + 1`. DeltaNet states and conv windows (48 layers): restored to their
   value after position `P + m` (the design below). The bonus becomes the next round's `x_{P'}` with
   `P' = P + m + 1`.
5. **Re-feed the MTP.** Its cache is truncated to `P` rows and the `m + 1` rows `(d_1, h_P), ..., (d_m, h_{P+m-1}),
   (bonus, h_{P+m})` are fed as one batch through the block; the last row's logits give the next round's
   `d_1`; `d_2..d_k` chain from the MTP's own hidden. `h_P..h_{P+m}` are rows `0..m` of the verify batch's final
   norm, which `forward_batch` keeps.

Stop tokens: the emitted sequence is checked in order and cut at the first stop id, exactly as the plain loop
does (it never emits past a stop, so neither does this).

`k = 0` (or no `--spec`) is the Phase 4 loop unchanged: `forward_token` per token.

## Why greedy `--spec` is lossless

Every emitted token is an `argmax` of the main model's logits at its position given exactly the tokens before
it, computed by the same arithmetic as the plain loop (the batched path is bit-identical to the sequential feed,
finding 41, and the rollback below restores bit-identical state). The draft only decides *which* positions get
computed in one batch, never *what* is emitted. So the ids are the plain loop's ids, and `tests/spec_identity.rs`
asserts it: 200 greedy tokens x 3 prompts x `k = 1..4`, resident and streamed on the tiny model; the real model
at resident and 8 GiB in `docs/data/spec_identity.txt`. A divergence is a bug.

## Rollback design: one snapshot plus replay from saved inputs (design "b without the weight pass")

The brief offered (a) `k + 1` snapshots of the 48-layer DeltaNet state, one per batch position, and (b) one
snapshot plus a replay of the accepted tokens, "almost certainly wrong on streamed rungs" because a replay through
the weights is a second disk pass.

Chosen: **(b′)**: one snapshot before the batch, and a replay of the accepted tokens through the DeltaNet
**recurrence only**, from inputs the verify batch saved, without touching a weight.

- What the recurrence of layer `L` at token `t` needs is exactly: the state after `t - 1`, the conv window after
  `t - 1`, the conv input row `mixed_t` (10,240 f32, the `attn_qkv` projection of the normed residual) and the two
  gate pre-activations `a_t`, `b_t` (48 f32 each). `mixed_t`, `a_t`, `b_t` are outputs of the batch's matmuls;
  `forward_batch` copies them into `SpecState::saved` per layer per row as it goes (the batch scratch is reused
  across layers, so the copy is necessary).
- Rollback to `m + 1` kept rows (`m + 1 < k + 1`): restore the snapshot (state and conv window of all 48 layers),
  then for `t = 0..m` run `causal_conv1d_step` and, per head, `l2norm`/scale/`gate_g`/`gate_beta`/`deltanet_step`
  on the saved rows: the very code the batch and the plain loop run, on the same f32 inputs, in the same order.
  The result is bit-identical to the state the batch had after row `m` and to what the plain loop would hold
  after consuming those tokens. When `m + 1 = k + 1` nothing is restored.
- The head loop is parallel over the 48 V heads on the pool, like the batch's.

Cost: the snapshot copy (150 MiB, `memcpy` at several GB/s: about 30 ms, once per round) plus per replayed token
the conv and the recurrence, the two stages finding 53 measured at 13.6 and 19.4 ms per token: about 33 ms per
replayed token, upper bound `k x 33 ms` per round, nothing from disk. Memory: one snapshot plus
`48 x (k + 1) x (10,240 + 96) x 4` bytes of saved rows (9.5 MiB at `k = 4`), against (a)'s `(k + 1) x 150 MiB`
(750 MiB at `k = 4`, three pinned layers at the 6 GiB rung). (b) as the brief states it (re-running the layers on
the accepted tokens) would read the streamed layers a second time per round and is not implemented; nothing needs
it, because the recurrence inputs are cheap to keep.

The KV caches truncate by length (`KvCache.len` and the two `Vec` lengths; capacity stays, so no allocation).
The conv window is part of the snapshot and of the replay.

## Memory: the plan lines `--spec k` adds (real model, per `docs/tiers.md` conventions, `max_pos` positions)

| line | formula | bytes at `k = 4`, `max_pos = 4096` |
|---|---|---|
| DeltaNet state snapshot | `48 x (48 x 128 x 128 + 10240 x 3) x 4` | 156,893,184 |
| saved recurrence inputs | `48 x (k + 1) x (10240 + 48 + 48) x 4` | 9,922,560 |
| verify batch scratch | `(k + 1)` rows of: residual x3, DeltaNet `mixed/z/a/b/conved/heads` and per-head `qn/kn/o`, attention `qg/k/v/q/gate/attn`, MLP `g/u/h`, four activation rows; plus `(k + 1) x vocab` logits and `(k + 1) x hidden` final-norm rows; the per-head score rows are the existing `AttnScratch` ones | formula in `tier::PlanInput::spec_bytes`, asserted against the live `SpecState::bytes()` on the tiny model in `tests/tier_plan.rs` |
| MTP KV cache | `2 x 4 x 256 x 4 = 8,192` per position `x max_pos` | 33,554,432 |
| MTP block scratch | one attention layer's `Scratch` (token and batch forms), the `eh_proj` input row, the draft logits | formula in `spec_bytes` |

The MTP arena (238,989,312 bytes) was already in the plan as "loaded, unused"; the label changes. At the 6 GiB
rung the additions are about 0.2 GB, i.e. one pinned layer fewer in the worst case (the plan headroom there was
71 to 159 MB); the ladder files the pinned count next to each `--spec` rung.

## Disk accounting (rule 5)

`forward_batch` acquires and releases each streamed layer exactly once, like `run_layers`, and asserts after the
pass that the consumed-byte delta equals the streamed spans (`check_pass`). The MTP block is always resident, so
drafting reads nothing from disk. `aqueduct run --spec k` asserts per round that the disk delta is exactly one
pass, and `tests/decode_alloc.rs` part (d) checks it on the tiny model with 7 of 9 layers streamed, together
with the allocation count (zero across 32 rounds, I/O thread included).

## Sampling with speculation (5.3): the acceptance rule

Let `p_i` be the target's processed distribution at batch row `i` (temperature, top-k, top-p, min-p applied,
renormalised) and `q_i` the draft's processed distribution for `d_{i+1}` (same settings, from the MTP logits),
from which `d_{i+1}` was drawn. For `i = 0..k-1`:

- accept `d_{i+1}` with probability `min(1, p_i(d_{i+1}) / q_i(d_{i+1}))` (a uniform draw `u < p/q`);
- on rejection, emit a token from the residual `norm(max(0, p_i - q_i))` and stop the round;
- if all `k` are accepted, sample the bonus from `p_k`.

This is the standard exact speculative sampling: the emitted token at each position is distributed as `p_i`
whatever `q_i` is. Greedy is the `temperature = 0` case of the same code path (`p`, `q` one-hot, so the rule is
the argmax comparison above). `tests/sample.rs` draws 20,000 tokens from a fixed 3-way `p` with a fixed 3-way
`q` through the acceptance rule and compares the histogram with the counts a plain draw from `p` gives, using a
chi-square statistic with 2 degrees of freedom against `p`'s expected counts: tolerance 13.82 (the 99.9 % point
of chi-square with 2 dof), stated in the test. The RNG is a seeded `xoshiro256**`; a run is reproducible per seed
at any thread count because the sampler runs on the calling thread and the logits are thread-invariant (Phase 3
contract).

## Where the speed comes from, and where it does not

On a streamed rung a round costs one disk pass (`bytes_disk / diskbw`, the same as one plain token) plus the
pinned layers' compute for `k + 1` activations, plus `k` drafts and the replay. It emits `m + 1` tokens. The
brief's cost model, `t_round ≈ bytes_ram / membw_eff + bytes_disk / diskbw + c_mtp x k + c_verify x (k + 1)`,
is filed against measurements per rung in `docs/ladder.md`; `c_verify` is the batched compute of one extra row
(the matmul with a cached row is compute-bound, finding 41: about 0.45 s per token over all 64 layers when
resident, proportionally less for the pinned share) and `c_mtp` about 45 ms (`docs/mtp.md`). Where the pinned
share is large (16 GiB, resident) the `k + 1` rows of compute are not hidden under anything, so the gain shrinks
and can invert; the measurement, not this paragraph, decides the default `k` (5.4).

## Measured (Phase 5.4; `docs/data/spec_acceptance.txt`, `docs/data/spec_identity_8g.txt`, `docs/ladder.md`)

- Identity: every `--spec` run equals its plain run: resident k = 1..5 on six prompts (30 of 30), 8 GiB k = 1..4 on
  the three fixture prompts (12 of 12), 200 greedy tokens each; on the tiny model with every acceptance count forced
  and the carried state compared after every round (`tests/spec_identity.rs`).
- Acceptance (resident, six prompts): 0.87 / 1.51 / 1.98 / 2.33 / 2.60 drafts accepted per round at k = 1..5, i.e.
  1.85 / 2.40 / 2.74 / 2.99 / 3.15 tokens per round; the first draft is accepted in about 80 % of rounds on every
  prompt, later positions fall off fast on prose (finding 67).
- Speed: where the disk is the bottleneck a round costs one disk pass plus little else, so the tokens per round
  become the speedup: 8 GiB 1.66 / 1.98 / 2.08 / 2.03 x at k = 1..4. Where it is not, each verify row is about 0.85 s
  of compute (the un-blocked kernel, finding 68): resident 1.06 / 0.98 / 0.94 / 0.86 / 0.76 x. The default `--spec`
  is k = 3 (the best streamed mean); plain decode stays the default when nothing is asked for.
- The round's parts (8 GiB, k = 3): verify 4.5 s (of which the plain pass is 3.6), MTP re-feed 0.12 to 0.18 s,
  chained draft 0.085 s per step, replay 0.03 to 0.09 s, snapshot 0.03 s.

## Phase 5.5: the blocked verify pass

The batched matmul that verifies a round is now the blocked GEMM (`matvec::matmul_t`, `docs/kquant-dot.md` Phase 5.5
section): each K-quant super-block is unpacked once and the `k + 1` rows are dotted against it, on every hardware
thread (a batched matmul is compute-bound and the sibling thread fills the ports its dependency chains leave idle;
the single-row matvec keeps the physical cores). Per row the arithmetic is the single-row kernel's bit for bit, so
nothing above this paragraph changes: identity holds at every `k` on the real model at resident (30 of 30) and at
8 GiB (12 of 12), `docs/data/spec_acceptance.txt`, `spec_identity_8g.txt`. What changes is the cost of a row:
on the streamed rungs the extra rows now add 0.16 s each to the disk pass (0.41 before), so `--spec 3` gives 2.27 x
at 5 GiB, 1.78 x at 11 GiB and 2.23 x at 8 GiB (finding 75, 76), and `--spec 4` edges `--spec 3` at 8 GiB; where
nothing hides the rows a round costs 2.35 plain tokens instead of 3.02 (the fitted `c_verify` is 0.53 s on a hot
resident rung, 0.45 plain tokens per row against 0.67), which makes the resident rung break-even (0.99 x) rather
than a loss (0.79 x). The reason it is not less is measured in finding 70: on this AVX2 core the int8
multiply-accumulate is the floor of a row, not the unpack. Plain decode stays the default and `--spec` keeps
`k = 3`.
