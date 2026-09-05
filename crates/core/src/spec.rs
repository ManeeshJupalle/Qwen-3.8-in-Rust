//! Speculative decoding (Phase 5.2, `docs/spec.md`): the MTP head drafts `k` tokens, one batched forward of
//! the `k + 1` tokens `[x_P, d_1..d_k]` verifies them (every weight row read once, streamed layers included),
//! the longest matching prefix plus the model's own next token are emitted, and the state is rolled back to
//! the last accepted position: KV caches by length, the DeltaNet states by restoring one snapshot taken before
//! the batch and replaying the accepted rows' conv and recurrence from the inputs the batch saved (no weight
//! is touched). Greedy verification is exact, so the emitted ids are the plain loop's ids bit for bit
//! (`tests/spec_identity.rs`); sampling uses the acceptance rule of `sample.rs`. Everything the round touches
//! is preallocated in `SpecState` (`tests/decode_alloc.rs` part d).

use std::time::Instant;

use crate::kernels::matvec::{matmul, matmul_fn, ActBuf, ActVec};
use crate::kernels::rmsnorm::rmsnorm;
use crate::layers::gqa_attention::KvCache;
use crate::layers::mtp::{MtpHead, MtpScratch};
use crate::layers::{BatchScratch, Mixer, MixerState};
use crate::model::{argmax, Model, ModelError, State};
use crate::sample::Sampler;

/// Counters of the speculative loop; `run --stats` files them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecStats {
    pub rounds: u64,
    /// Draft tokens verified (`k` per round) and accepted.
    pub drafted: u64,
    pub accepted: u64,
    /// Tokens emitted by the rounds (accepted drafts plus the bonus / resampled token).
    pub emitted: u64,
    /// `accepted_at[i]`: rounds in which draft `i + 1` was accepted.
    pub accepted_at: Vec<u64>,
    /// `hist[m]`: rounds that accepted exactly `m` drafts.
    pub hist: Vec<u64>,
    pub snapshot_ns: u64,
    pub verify_ns: u64,
    pub replay_ns: u64,
    /// The MTP re-feed of the accepted rows (block on `m + 1` rows, head once) and the chained single-row
    /// drafts (block and head each), with their counts.
    pub refeed_ns: u64,
    pub refeeds: u64,
    pub chain_ns: u64,
    pub chain_steps: u64,
}

impl SpecStats {
    pub fn mean_accepted(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.accepted as f64 / self.rounds as f64
        }
    }
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

/// Everything speculation needs, sized once (`Model::new_state_spec`): the verification batch's buffers, the
/// rollback snapshot and saved inputs, the MTP's cache and scratch, the drafts and their distributions.
pub struct SpecState {
    pub k: usize,
    pub batch: BatchScratch,
    /// Residual ping-pong and the post-final-norm rows of the batch (`k + 1` rows each).
    pub h0: Vec<f32>,
    pub h1: Vec<f32>,
    pub normed: Vec<f32>,
    pub lm_acts: Vec<ActBuf>,
    /// `(k + 1) x vocab`.
    pub logits: Vec<f32>,
    /// The DeltaNet states and conv windows before the batch (rec then conv per layer, flat).
    pub snapshot: Vec<f32>,
    /// Per DeltaNet layer, per batch row: the conv input row and the two gate pre-activations.
    pub saved_mixed: Vec<f32>,
    pub saved_a: Vec<f32>,
    pub saved_b: Vec<f32>,
    conv_dim: usize,
    n_v: usize,
    pub mtp_cache: MixerState,
    pub mtp: MtpScratch,
    /// `k` draft tokens for the next round, and their dense draft distributions (sampling only).
    pub drafts: Vec<u32>,
    pub q_dense: Vec<f32>,
    pub batch_ids: Vec<u32>,
    pub emitted: Vec<u32>,
    pub stats: SpecStats,
}

impl SpecState {
    fn new(model: &Model, k: usize) -> Result<SpecState, ModelError> {
        let mtp = model.mtp.as_ref().ok_or(ModelError::NoMtp)?;
        assert!(k >= 1, "spec: k must be at least 1");
        let t = k + 1;
        let hidden = model.hidden();
        let vocab = model.vocab();
        let (conv_dim, n_v) = (model.cfg.dn_qkv_dim as usize, model.cfg.dn_n_v_heads as usize);
        let n_dn = model.cfg.deltanet_layers.len();
        let snap_len = n_dn * (model.cfg.dn_state_elems_per_layer + model.cfg.dn_conv_state_elems_per_layer) as usize;
        Ok(SpecState {
            k,
            batch: BatchScratch::for_layers(&model.layers, hidden, t),
            h0: vec![0f32; t * hidden],
            h1: vec![0f32; t * hidden],
            normed: vec![0f32; t * hidden],
            lm_acts: (0..t).map(|_| ActBuf::with_capacity(hidden)).collect(),
            logits: vec![0f32; t * vocab],
            snapshot: vec![0f32; snap_len],
            saved_mixed: vec![0f32; n_dn * t * conv_dim],
            saved_a: vec![0f32; n_dn * t * n_v],
            saved_b: vec![0f32; n_dn * t * n_v],
            conv_dim,
            n_v,
            mtp_cache: mtp.new_cache(),
            mtp: mtp.new_scratch(vocab, t),
            drafts: Vec::with_capacity(k),
            q_dense: vec![0f32; k * vocab],
            batch_ids: Vec::with_capacity(t),
            emitted: Vec::with_capacity(t),
            stats: SpecStats { accepted_at: vec![0; k], hist: vec![0; t], ..Default::default() },
        })
    }

    /// Size the MTP cache and its score rows for `max_pos` rows.
    pub fn reserve(&mut self, max_pos: usize) {
        if let MixerState::Attention(c) = &mut self.mtp_cache {
            c.reserve(max_pos);
        }
        self.mtp.reserve(max_pos);
    }

    /// Bytes held (capacities); `tier::PlanInput::spec_bytes` is the formula (`tests/tier_plan.rs`).
    pub fn bytes(&self) -> usize {
        let cache = match &self.mtp_cache {
            MixerState::Attention(c) => c.bytes(),
            MixerState::DeltaNet(d) => d.bytes(),
        };
        self.batch.bytes()
            + (self.h0.capacity() + self.h1.capacity() + self.normed.capacity() + self.logits.capacity() + self.snapshot.capacity() + self.saved_mixed.capacity() + self.saved_a.capacity() + self.saved_b.capacity() + self.q_dense.capacity()) * 4
            + self.lm_acts.iter().map(|a| a.bytes()).sum::<usize>()
            + cache
            + self.mtp.bytes()
            + (self.drafts.capacity() + self.batch_ids.capacity() + self.emitted.capacity()) * 4
            + (self.stats.accepted_at.capacity() + self.stats.hist.capacity()) * 8
    }

    /// The MTP's rows so far.
    pub fn mtp_rows(&self) -> usize {
        MtpHead::rows(&self.mtp_cache)
    }
}

fn snapshot(layers: &[MixerState], into: &mut [f32]) {
    let mut off = 0;
    for l in layers {
        if let MixerState::DeltaNet(d) = l {
            into[off..off + d.rec.len()].copy_from_slice(&d.rec);
            off += d.rec.len();
            into[off..off + d.conv.len()].copy_from_slice(&d.conv);
            off += d.conv.len();
        }
    }
    assert_eq!(off, into.len());
}

fn restore(layers: &mut [MixerState], from: &[f32]) {
    let mut off = 0;
    for l in layers {
        if let MixerState::DeltaNet(d) = l {
            let (nr, nc) = (d.rec.len(), d.conv.len());
            d.rec.copy_from_slice(&from[off..off + nr]);
            off += nr;
            d.conv.copy_from_slice(&from[off..off + nc]);
            off += nc;
        }
    }
    assert_eq!(off, from.len());
}

impl Model {
    /// A state with the speculative buffers for `k` drafts per round. `reserve(max_pos)` must cover the
    /// prompt, the tokens to generate and `k + 1` more (a batch consumes `k + 1` positions before rollback).
    pub fn new_state_spec(&self, k: usize) -> Result<State, ModelError> {
        let mut s = self.new_state();
        s.spec = Some(Box::new(SpecState::new(self, k)?));
        Ok(s)
    }

    /// One MTP draft from the logits in `spec.mtp.logits`: greedy argmax, or a draw from the processed
    /// distribution, which is also written densely to `q_dense[i]` for the acceptance rule.
    fn draft_token(&self, spec: &mut SpecState, sampler: &mut Sampler, i: usize) -> u32 {
        if sampler.cfg.is_greedy() {
            argmax(&spec.mtp.logits)
        } else {
            sampler.process(&spec.mtp.logits);
            let d = sampler.draw();
            let vocab = self.vocab();
            sampler.write_dense(&mut spec.q_dense[i * vocab..(i + 1) * vocab]);
            d
        }
    }

    /// Chain drafts `from..k` from the MTP hidden of row `prev` of `spec.mtp.hout` and the draft `spec.drafts[from - 1]`.
    fn chain_drafts(&self, spec: &mut SpecState, sampler: &mut Sampler, mut prev: usize, from: usize) {
        let mtp = self.mtp.as_ref().expect("mtp");
        let t0 = Instant::now();
        let mut steps = 0u64;
        for i in from..spec.k {
            let last = spec.drafts[i - 1];
            spec.mtp.take_prev(prev);
            mtp.set_row_with(&mut spec.mtp, 0, None, &|e| self.embed_token_into(last, e));
            mtp.rows_in(1, &mut spec.mtp_cache, &mut spec.mtp, self.threads);
            mtp.head_in(&self.lm_head, &mut spec.mtp, 0, self.threads);
            let d = self.draft_token(spec, sampler, i);
            spec.drafts.push(d);
            prev = 0;
            steps += 1;
        }
        spec.stats.chain_ns += t0.elapsed().as_nanos() as u64;
        spec.stats.chain_steps += steps;
    }

    /// Prefill for speculation (prompt time, allocating like `prefill`): the batched prefill of `ids`, the
    /// first token (argmax, or a draw), the MTP fed with the prompt's `T` rows `(x_{j+1}, h_j)` as one batch
    /// (`docs/mtp.md`), and the first `k` drafts. Returns `x_T`, the token the caller emits and then passes to
    /// the first `spec_round`. On a state that already consumed `P` positions (a later chat turn) the rows are
    /// `(x_{P+j+1}, h_{P+j})` at positions `P..`, after the MTP's draft rows from the last round are forgotten.
    pub fn spec_prefill(&self, ids: &[u32], state: &mut State, sampler: &mut Sampler) -> u32 {
        let mtp = self.mtp.as_ref().expect("spec_prefill: the model has no MTP head");
        let t = ids.len();
        let hidden = self.hidden();
        let p = state.pos as usize;
        let hs = self.prefill(ids, state);
        let mut normed = vec![0f32; t * hidden];
        for i in 0..t {
            rmsnorm(&hs[i * hidden..(i + 1) * hidden], &self.output_norm, self.cfg.rms_norm_eps, &mut normed[i * hidden..(i + 1) * hidden]);
        }
        let mut logits = vec![0f32; self.vocab()];
        self.logits_of(&hs[(t - 1) * hidden..], &mut logits);
        let x_t = if sampler.cfg.is_greedy() { argmax(&logits) } else { sampler.sample(&logits) };

        // the MTP over the prompt: row j = (x_{j+1}, h_j), x_T = the token just chosen; a used state's MTP has
        // rows 0..P-1 of target-based pairs plus the last round's chained draft rows, which go
        let spec = state.spec.as_mut().expect("spec_prefill: state without spec buffers");
        assert!(spec.mtp_rows() >= p, "spec_prefill: the MTP has {} rows for {p} consumed positions", spec.mtp_rows());
        MtpHead::truncate(&mut spec.mtp_cache, p);
        let mut cats = vec![0f32; t * 2 * hidden];
        let mut e = vec![0f32; hidden];
        for j in 0..t {
            let x = if j + 1 < t { ids[j + 1] } else { x_t };
            self.embed_token_into(x, &mut e);
            let row = &mut cats[j * 2 * hidden..(j + 1) * 2 * hidden];
            rmsnorm(&e, &mtp.enorm, mtp.eps, &mut row[..hidden]);
            rmsnorm(&normed[j * hidden..(j + 1) * hidden], &mtp.hnorm, mtp.eps, &mut row[hidden..]);
        }
        let acts: Vec<ActVec<'_>> = cats.chunks_exact(2 * hidden).map(ActVec::new).collect();
        let et = mtp.eh_proj.w.ggml_type;
        let a: Vec<_> = acts.iter().map(|x| x.act_for(et)).collect();
        let mut u = vec![0f32; t * hidden];
        matmul(&mtp.eh_proj.w, &a, &mut u, self.threads);
        let mut y = vec![0f32; t * hidden];
        mtp.block.forward_prefill(&u, t, p as u32, &mut spec.mtp_cache, &mut y, self.threads);
        // the last row's post-norm hidden into the scratch, its logits, draft 1, then the chain
        rmsnorm(&y[(t - 1) * hidden..], &mtp.head_norm, mtp.eps, &mut spec.mtp.hout[..hidden]);
        mtp.head_in(&self.lm_head, &mut spec.mtp, 0, self.threads);
        spec.drafts.clear();
        let d1 = self.draft_token(spec, sampler, 0);
        spec.drafts.push(d1);
        self.chain_drafts(spec, sampler, 0, 1);
        x_t
    }

    /// The verification batch: `spec.batch_ids[..t]` at the state's position through every layer (each
    /// streamed layer read once), the DeltaNet inputs saved per layer and row, the post-final-norm rows in
    /// `spec.normed` and the logits of every row in `spec.logits`. Allocation-free.
    fn forward_batch(&self, state: &mut State, t: usize) {
        let State { layers, pos, scratch, spec, .. } = state;
        let spec = spec.as_mut().expect("forward_batch: no spec buffers");
        let hidden = self.hidden();
        let vocab = self.vocab();
        let big_t = spec.k + 1;
        let (cd, nv) = (spec.conv_dim, spec.n_v);
        assert!(t <= big_t);
        let before = self.consumed();
        for i in 0..t {
            self.embed_token_into(spec.batch_ids[i], &mut spec.h0[i * hidden..(i + 1) * hidden]);
        }
        let mut dn_i = 0usize;
        for (li, (layer, st)) in self.layers.iter().zip(layers.iter_mut()).enumerate() {
            self.acquire(li);
            layer.forward_batch_in(&spec.h0[..t * hidden], t, *pos, st, &mut spec.batch, scratch, &mut spec.h1[..t * hidden], self.threads);
            self.release(li);
            if let Mixer::DeltaNet(_) = &layer.mixer {
                let dnb = spec.batch.dn.as_ref().expect("batch dn");
                let base = dn_i * big_t;
                spec.saved_mixed[base * cd..(base + t) * cd].copy_from_slice(&dnb.mixed[..t * cd]);
                spec.saved_a[base * nv..(base + t) * nv].copy_from_slice(&dnb.a[..t * nv]);
                spec.saved_b[base * nv..(base + t) * nv].copy_from_slice(&dnb.b[..t * nv]);
                dn_i += 1;
            }
            std::mem::swap(&mut spec.h0, &mut spec.h1);
        }
        *pos += t as u32;
        self.check_pass(before);
        let ht = self.lm_head.ggml_type;
        let SpecState { h0, normed, lm_acts, logits, .. } = &mut **spec;
        for i in 0..t {
            rmsnorm(&h0[i * hidden..(i + 1) * hidden], &self.output_norm, self.cfg.rms_norm_eps, &mut normed[i * hidden..(i + 1) * hidden]);
            lm_acts[i].fill(&normed[i * hidden..(i + 1) * hidden], &[ht]);
        }
        let normed: &[f32] = normed;
        matmul_fn(&self.lm_head, t, &|i| lm_acts[i].act_for(ht, &normed[i * hidden..(i + 1) * hidden]), &mut logits[..t * vocab], self.threads);
    }

    /// Row `i` of the batch logits of the last round.
    pub fn spec_logits_row<'a>(&self, state: &'a State, i: usize) -> &'a [f32] {
        let vocab = self.vocab();
        let spec = state.spec.as_ref().expect("spec");
        &spec.logits[i * vocab..(i + 1) * vocab]
    }

    /// One round (`docs/spec.md`): verify `[x_p, d_1..d_k]`, accept, roll back, re-feed the MTP and draft the
    /// next round. The emitted tokens (`m` accepted drafts, then the bonus or resampled token) are in
    /// `state.spec.emitted`; the last of them is the next round's `x_p`. Returns `m`. Allocation-free.
    pub fn spec_round(&self, state: &mut State, x_p: u32, sampler: &mut Sampler) -> usize {
        let mtp = self.mtp.as_ref().expect("spec_round: the model has no MTP head");
        let hidden = self.hidden();
        let vocab = self.vocab();
        let k = state.spec.as_ref().expect("spec").k;
        let t = k + 1;
        let p = state.pos as usize;
        {
            let spec = state.spec.as_mut().unwrap();
            assert_eq!(spec.drafts.len(), k, "spec_round: {} drafts for k = {k}", spec.drafts.len());
            assert_eq!(spec.mtp_rows(), p + k - 1, "spec_round: MTP rows {} at position {p} (k {k})", spec.mtp_rows());
            spec.batch_ids.clear();
            spec.batch_ids.push(x_p);
            for i in 0..k {
                spec.batch_ids.push(spec.drafts[i]);
            }
            let t0 = Instant::now();
            snapshot(&state.layers, &mut spec.snapshot);
            spec.stats.snapshot_ns += t0.elapsed().as_nanos() as u64;
        }
        let t0 = Instant::now();
        self.forward_batch(state, t);
        let State { layers, pos, spec, .. } = state;
        let spec = spec.as_mut().unwrap();
        spec.stats.verify_ns += t0.elapsed().as_nanos() as u64;

        // ---- accept
        let mut m = 0usize;
        spec.emitted.clear();
        if sampler.cfg.is_greedy() {
            while m < k {
                let a = argmax(&spec.logits[m * vocab..(m + 1) * vocab]);
                spec.emitted.push(a);
                if a != spec.drafts[m] {
                    break;
                }
                m += 1;
            }
            if m == k {
                spec.emitted.push(argmax(&spec.logits[k * vocab..(k + 1) * vocab]));
            }
        } else {
            while m < k {
                let (ok, tok) = sampler.accept_or_resample(&spec.logits[m * vocab..(m + 1) * vocab], &spec.q_dense[m * vocab..(m + 1) * vocab], spec.drafts[m]);
                spec.emitted.push(tok);
                if !ok {
                    break;
                }
                m += 1;
            }
            if m == k {
                sampler.process(&spec.logits[k * vocab..(k + 1) * vocab]);
                spec.emitted.push(sampler.draw());
            }
        }
        let keep = m + 1;
        debug_assert_eq!(spec.emitted.len(), keep);
        {
            let s = &mut spec.stats;
            s.rounds += 1;
            s.drafted += k as u64;
            s.accepted += m as u64;
            for i in 0..m {
                s.accepted_at[i] += 1;
            }
            s.hist[m] += 1;
            s.emitted += keep as u64;
        }

        // ---- roll back to `keep` consumed rows
        if keep < t {
            let t0 = Instant::now();
            restore(layers, &spec.snapshot);
            let (cd, nv) = (spec.conv_dim, spec.n_v);
            let dnb = spec.batch.dn.as_mut().expect("batch dn");
            let mut dn_i = 0usize;
            for (layer, st) in self.layers.iter().zip(layers.iter_mut()) {
                match (&layer.mixer, st) {
                    (Mixer::DeltaNet(d), MixerState::DeltaNet(s)) => {
                        let base = dn_i * t;
                        d.replay_in(&spec.saved_mixed[base * cd..(base + keep) * cd], &spec.saved_a[base * nv..(base + keep) * nv], &spec.saved_b[base * nv..(base + keep) * nv], keep, s, dnb, self.threads);
                        dn_i += 1;
                    }
                    (Mixer::Attention(_), MixerState::Attention(c)) => c.truncate(p + keep),
                    _ => unreachable!(),
                }
            }
            *pos = (p + keep) as u32;
            spec.stats.replay_ns += t0.elapsed().as_nanos() as u64;
        }

        // ---- the MTP: forget its draft rows, feed the accepted pairs, draft the next round
        let t0 = Instant::now();
        MtpHead::truncate(&mut spec.mtp_cache, p);
        for i in 0..keep {
            let x = spec.emitted[i];
            mtp.set_row_with(&mut spec.mtp, i, Some(&spec.normed[i * hidden..(i + 1) * hidden]), &|e| self.embed_token_into(x, e));
        }
        mtp.rows_in(keep, &mut spec.mtp_cache, &mut spec.mtp, self.threads);
        mtp.head_in(&self.lm_head, &mut spec.mtp, keep - 1, self.threads);
        spec.drafts.clear();
        let d1 = self.draft_token(spec, sampler, 0);
        spec.drafts.push(d1);
        spec.stats.refeed_ns += t0.elapsed().as_nanos() as u64;
        spec.stats.refeeds += 1;
        self.chain_drafts(spec, sampler, keep - 1, 1);
        m
    }

    /// Replace the drafts for the next round (tests: force a known acceptance count).
    pub fn spec_set_drafts(&self, state: &mut State, drafts: &[u32]) {
        let spec = state.spec.as_mut().expect("spec");
        assert_eq!(drafts.len(), spec.k);
        spec.drafts.clear();
        spec.drafts.extend_from_slice(drafts);
        if !spec.q_dense.is_empty() {
            // a forced draft is a point mass for the acceptance rule
            let vocab = self.vocab();
            for (i, &d) in drafts.iter().enumerate() {
                let q = &mut spec.q_dense[i * vocab..(i + 1) * vocab];
                q.fill(0.0);
                q[d as usize] = 1.0;
            }
        }
    }

    /// The MTP cache of a spec state (tests).
    pub fn spec_mtp_cache(state: &State) -> &KvCache {
        MtpHead::cache_of(&state.spec.as_ref().expect("spec").mtp_cache)
    }
}
