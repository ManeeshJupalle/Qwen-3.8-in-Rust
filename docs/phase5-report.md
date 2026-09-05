# Phase 5 report (speculation, sampling, chat)

Commits 01a5993 (`docs/mtp.md`, `docs/spec.md`, before the code), 98353cd (the code and tests) and the ones
after them on `main`, 2026-09-04 / 05. Design in `docs/mtp.md` and `docs/spec.md`; numbers in `docs/data/`
(`spec_acceptance.txt`, `spec_identity_8g.txt`, `ladder.txt`, `spec_cost_model.txt`, `chat_16g.txt`) and
`docs/ladder.md`; findings 62 to 69 in `docs/payload-vs-doc.md`. Machine as in Phase 4 (i7-9750H, 32 GB,
NVMe, Windows 11), on this run mostly in its **throttled** state (finding 40: the plain resident token measured
1.21 to 1.41 s against 0.74 s in the Phase 4 ladder taken cool the same morning; membw 27.6 GB/s at the start of
the 8 GiB sweep against 30.1 in `doctor`). Every speedup below is against a plain run taken in the same block of
minutes, and the identity gates do not depend on the clock at all.

```
PHASE 5 REPORT
tests          : 74/74 non-ignored (22 unit + 52 integration over 21 targets: Phase 4's 65 + mtp_oracle 1, spec_identity 2,
                 chat_template 1, sample 1, the core sample/chat unit tests 4; decode_alloc part d and tier_plan extended) + 2
                 ignored model-scale gates unchanged; clippy --all-targets clean
mtp            : layer type GQA (attn_q/k/v/output, q_norm/k_norm; no ssm_*); consumes embed of the token that follows position j
                 (shared token_embd) and the target's POST-final-norm hidden at j (what the lm_head read; vLLM, SGLang and
                 llama.cpp all serve that), each through its own RMSNorm (enorm / hnorm, +1 folded in by the converter),
                 concatenated embed-first into eh_proj; then one attention block, shared_head_norm, the shared output.weight;
                 chained rows take the MTP's post-norm hidden; cache: its own KV cache, row j at position j, truncated to the
                 consumed count after every round; prefill-fill yes (the prompt's T rows as one batch, draft 1 from the last);
                 oracle 60/60 (draft-1 logits at 0.00 of budget on all 3 prompts, top-1 logit at step 20 within 1e-7)
rollback       : design b' (one snapshot before the batch + replay of the accepted rows' conv and recurrence from the batch's saved
                 conv inputs and gate pre-activations; no weight pass); snapshot bytes per position n/a: one snapshot of 156.9 MB
                 for the round, plus 1.98 MB of saved inputs per position (48 x 41,344 B); plan delta at 6 GiB 202.6 MB at k=1,
                 218.6 MB at k=4 (snapshot 156.9 + MTP KV 33.6 at 4096 positions + scratch); pinned 10 instead of 11 layers
identity       : spec vs no-spec 200-token ids: resident k=1 6/6 k=2 6/6 k=3 6/6 k=4 6/6 k=5 6/6 (all six prompts; the three
                 fixture prompts are 3/3 at every k); 8 GiB k=1 3/3 k=2 3/3 k=3 3/3 k=4 3/3 (200 tokens, 22 pinned / 42 streamed,
                 one disk pass per round asserted every round); tiny model: 200 tokens x 3 prompts x
                 k=1..4 resident and streamed identical, with every acceptance count 0..k forced and the whole carried state equal
                 to a plain state after every round
acceptance     : mean accepted/round at k=1..5 (resident, 6 prompts, greedy): 0.87 1.51 1.98 2.33 2.60 (per-draft rate 87 75 66 58
                 52 %; by position 0.85 / 0.81 0.59 / 0.79 0.55 0.41 / 0.78 0.56 0.39 0.26 / 0.79 0.54 0.37 0.25 0.19; tokens per round
                 1.85 2.40 2.74 2.99 3.15); at 8 GiB (3 fixture prompts) 0.87 1.53 2.01 2.40 for k=1..4 and 1.66 x 1.98 x 2.08 x 2.03 x
                 over the plain 3.635 s/token; chosen k=3 because it is the fastest mean on the streamed rung (2.08 x; k=4 gains
                 nothing more since the 4th draft is accepted in 25 % of rounds and costs a chained MTP step + a verify row) while
                 losing least where the disk is not the bottleneck (0.94 x resident: every extra verify row is 0.85 s of compute
                 there, finding 68; at resident --spec 1 is the only setting that does not lose, 1.06 x, and plain is the default)
ladder (spec)  : 5 GiB (8 GB laptop's free RAM)  8 pinned  4.316 -> 2.028 s/token  2.13 x  acceptance 63 % (1.89/round, 2.87 tokens/round)
                 6 GiB                            12 pinned 4.095 -> 1.979          2.07 x  63 %
                 8 GiB                            21 pinned 3.628 -> 1.812          2.00 x  63 %
                 11 GiB (16 GB laptop's free RAM) 35 pinned 2.790 -> 1.814          1.54 x  63 %
                 12 GiB                           39 pinned 2.690 -> 2.050          1.31 x  63 %
                 16 GiB                           57 pinned 1.155 -> 1.129          1.02 x  63 %
                 resident                         64 pinned 0.830 -> 1.057          0.79 x  63 %
                 (32 greedy tokens x 3 prompts; ids identical at every rung, plain and spec, and equal to Phase 3; peak RSS under
                 every cap with the spec buffers; 5 to 12 GiB in the mildly throttled state, 16 GiB and resident cool; the plain
                 cost model's worst ratio 1.36 at resident, every rung within 2 x)
sampling       : defaults from generation_config: do_sample=true, temperature=1.0, top_k=20, top_p=0.95 (no min_p, no repetition
                 penalty: none implemented, a file naming one is refused); histogram test chi2 4.66 (q != p), 0.95 (q == p),
                 1.10 (point-mass draft) vs tol 13.82 (99.9 % of chi-square(2)), plain sampler 2.23; seeded runs reproduce
chat           : template 7/7 byte-identical (2 doc examples + 5 multi-turn, ids identical too); multi-turn at 16 GiB (--spec 3,
                 sampling from generation_config, thinking on, 55 pinned / 9 streamed): three turns, 0.64 and 0.72 tok/s on the
                 two turns with a clean clock (the first turn spans a laptop shutdown: the engine resumed and answered correctly),
                 acceptance 66.7 / 58.0 / 62.3 % (2.00 / 1.74 / 1.87 accepted per round); the conversation prefix was reused on
                 turns 2 and 3 (no re-encoding), turn 3 quoted turn 1's question exactly; docs/data/chat_16g.txt
cost model     : c_mtp 83 ms (chained draft step, 64 ms cool), c_verify 558 ms per extra row (fitted at resident); predicted vs
                 measured s/round 6.80/5.39 6.58/5.26 6.11/4.82 5.27/4.82 5.17/5.45 3.64/3.00 3.31/2.81 at 5/6/8/11/12/16 GiB/resident,
                 ratios 0.79 0.80 0.79 0.91 1.05 0.82 0.85 (the sum overstates the streamed rungs by ~20 %: the rows' compute hides
                 under the disk; docs/data/spec_cost_model.txt, tools/spec_cost_model.py)
findings       : 62 the MTP consumes the post-final-norm hidden (llama.cpp's comment says the opposite of its code); 63 the
                 converter's +1 reaches the nextn norms (proved on the tiny GGUF); 64 the head is a GQA block with its own
                 cache, row j = (x_{j+1}, h_j), filled over the prompt; 65 rollback needs one snapshot + 10 MB, not k+1
                 snapshots; 66 minijinja needs raise_exception, a Python tojson and the string methods to match HF byte for
                 byte; 67 acceptance is a property of the text (90 % on code and repetitive continuations, 25 to 70 % on prose);
                 68 a verify row costs 0.56 to 0.85 s of compute, so speculation pays only where the disk is the bottleneck (a
                 t-way blocked kernel is the fix); 69 the ladder: 2.1 x at 5 to 8 GiB, break-even at 16 GiB, 0.79 x resident
blocked on     : nothing
did NOT do     : a t-way blocked verify kernel (the batched matmul re-unpacks each weight row per activation, finding 68; it
                 is what would make --spec pay at 16 GiB and resident); an acceptance sweep with sampling on the real model
                 (the sweep is greedy; sampled speculation is exercised by the unit histogram test and by `chat`); the
                 pre-norm-vs-post-norm MTP input experiment (the three references agree, so it was not run); the Unsloth
                 Q6_K/Q8_0 MTP copy against bartowski's Q4_0 (how much of the prose acceptance gap is quantisation);
                 tool-call rendering in the template fixtures (`tojson` is implemented, no tool case is filed); the Linux
                 build's Ctrl-C handler and streaming path (compiled, unrun, as in Phase 4); a ladder on a cool machine (the
                 box throttled through the session: rungs 8 GiB and up were re-run after a 10-minute idle and the two states
                 are both filed); `--spec` picking its own k from the plan (k is a constant; the report says where it loses);
                 the `disk_bytes_per_token` stat when a stop token ends a run (it counts one pass more than the steps it
                 divides by, an overstatement of 1/steps; the ladder runs have no stop)
```

