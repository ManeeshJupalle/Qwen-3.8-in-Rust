//! Sampling (Phase 5.3): greedy, temperature, top-k, top-p, min-p, seeded; and the exact speculative
//! acceptance rule (`docs/spec.md`). The processing order is HF `generate`'s warper order (temperature, top-k,
//! top-p, min-p), each on the distribution the previous one left. There is no repetition penalty:
//! `generation_config.json` names none (`tools/model_facts.py` table), and a value it does not give is not
//! invented; a file that names one is refused rather than silently ignored.
//!
//! Allocation-free after construction: the candidate index and probability buffers are `vocab` long and the
//! kept set is a prefix of them. The RNG is xoshiro256** seeded through splitmix64 (the reference
//! constructions of Blackman and Vigna), so a run is reproducible per seed; the sampler runs on the calling
//! thread and the logits are thread-invariant (Phase 3 contract), so the thread count does not enter.

use std::fmt;

/// xoshiro256** (Blackman & Vigna 2018), seeded by splitmix64 from a u64.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed(seed: u64) -> Rng {
        let mut x = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            // splitmix64
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = z ^ (z >> 31);
        }
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 24 random bits (exact in f32).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
}

/// The generation settings. `temperature <= 0` is greedy (argmax; the other fields are then unused).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    pub temperature: f32,
    /// 0 = off.
    pub top_k: usize,
    /// `>= 1` = off.
    pub top_p: f32,
    /// 0 = off.
    pub min_p: f32,
    pub seed: u64,
}

impl SamplerConfig {
    pub const fn greedy() -> SamplerConfig {
        SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0 }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Defaults from a `generation_config.json`: `temperature`, `top_k`, `top_p`, `min_p` if present; `do_sample:
    /// false` makes the result greedy. Returns the config and the list of fields the file actually gave (the
    /// report records them). A `repetition_penalty` other than 1 is refused: nothing implements it.
    pub fn from_generation_config_json(text: &str) -> Result<(SamplerConfig, Vec<String>), SampleError> {
        let v = json::parse(text).map_err(SampleError::Json)?;
        let obj = match &v {
            json::Json::Obj(o) => o,
            _ => return Err(SampleError::Json("generation_config.json: top level is not an object".into())),
        };
        let get = |k: &str| obj.iter().find(|(kk, _)| kk == k).map(|(_, v)| v);
        let num = |k: &str| -> Result<Option<f64>, SampleError> {
            match get(k) {
                None => Ok(None),
                Some(json::Json::Num(n)) => Ok(Some(*n)),
                Some(_) => Err(SampleError::Json(format!("generation_config.json: {k} is not a number"))),
            }
        };
        let mut cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0 };
        let mut given = Vec::new();
        if let Some(t) = num("temperature")? {
            cfg.temperature = t as f32;
            given.push(format!("temperature={t}"));
        }
        if let Some(k) = num("top_k")? {
            cfg.top_k = k as usize;
            given.push(format!("top_k={k}"));
        }
        if let Some(p) = num("top_p")? {
            cfg.top_p = p as f32;
            given.push(format!("top_p={p}"));
        }
        if let Some(p) = num("min_p")? {
            cfg.min_p = p as f32;
            given.push(format!("min_p={p}"));
        }
        if let Some(r) = num("repetition_penalty")? {
            if (r - 1.0).abs() > 1e-9 {
                return Err(SampleError::Unsupported(format!("repetition_penalty={r} is named by generation_config.json and not implemented")));
            }
            given.push(format!("repetition_penalty={r} (identity)"));
        }
        match get("do_sample") {
            Some(json::Json::Bool(false)) => {
                cfg.temperature = 0.0;
                given.push("do_sample=false (greedy)".into());
            }
            Some(json::Json::Bool(true)) => given.push("do_sample=true".into()),
            _ => {}
        }
        Ok((cfg, given))
    }
}

impl fmt::Display for SamplerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_greedy() {
            write!(f, "greedy (temperature 0)")
        } else {
            write!(f, "temperature {} top_k {} top_p {} min_p {} seed {}", self.temperature, self.top_k, self.top_p, self.min_p, self.seed)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SampleError {
    #[error("{0}")]
    Json(String),
    #[error("{0}")]
    Unsupported(String),
}

/// The sampler: a config, an RNG, and the buffers of the processed distribution (a kept set of `n` candidates,
/// ids in `idx[..n]`, probabilities in `probs[..n]`, sorted by probability descending, summing to 1).
pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: Rng,
    idx: Vec<u32>,
    probs: Vec<f32>,
    n: usize,
    vocab: usize,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig, vocab: usize) -> Sampler {
        let rng = Rng::seed(cfg.seed);
        Sampler { cfg, rng, idx: vec![0u32; vocab], probs: vec![0f32; vocab], n: 0, vocab }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.cfg.seed = seed;
        self.rng = Rng::seed(seed);
    }

    /// Bytes held, for the memory plan.
    pub fn bytes(&self) -> usize {
        self.idx.capacity() * 4 + self.probs.capacity() * 4
    }

    /// The kept set of the last `process`: `(ids, probs)`.
    pub fn kept(&self) -> (&[u32], &[f32]) {
        (&self.idx[..self.n], &self.probs[..self.n])
    }

    /// The processed distribution of `logits`, into the internal buffers. Greedy: the argmax alone.
    pub fn process(&mut self, logits: &[f32]) -> (&[u32], &[f32]) {
        assert_eq!(logits.len(), self.vocab, "sampler: logits length");
        let cfg = &self.cfg;
        if cfg.is_greedy() {
            self.idx[0] = crate::model::argmax(logits);
            self.probs[0] = 1.0;
            self.n = 1;
            return self.kept();
        }
        let vocab = self.vocab;
        for (i, v) in self.idx.iter_mut().enumerate() {
            *v = i as u32;
        }
        // candidates: the top-k by logit (ties: the lower id first, as the sort below is stable in that order),
        // or the whole vocabulary, sorted descending
        let n_cand = if cfg.top_k > 0 && cfg.top_k < vocab { cfg.top_k } else { vocab };
        let by_logit_desc = |a: &u32, b: &u32| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
        if n_cand < vocab {
            self.idx.select_nth_unstable_by(n_cand - 1, by_logit_desc);
        }
        self.idx[..n_cand].sort_unstable_by(by_logit_desc);
        // softmax of the candidates at the temperature
        let inv_t = 1.0 / cfg.temperature;
        let m = logits[self.idx[0] as usize] * inv_t;
        let mut sum = 0f32;
        for i in 0..n_cand {
            let e = (logits[self.idx[i] as usize] * inv_t - m).exp();
            self.probs[i] = e;
            sum += e;
        }
        for p in &mut self.probs[..n_cand] {
            *p /= sum;
        }
        // top-p: keep a candidate while the mass strictly above it is below top_p (HF's rule read in
        // descending order: the token that crosses the threshold stays)
        let mut n = n_cand;
        if cfg.top_p < 1.0 {
            let mut cum = 0f32;
            n = 0;
            while n < n_cand && cum < cfg.top_p {
                cum += self.probs[n];
                n += 1;
            }
            n = n.max(1);
        }
        // min-p: drop candidates below min_p x the top probability (at least one stays)
        if cfg.min_p > 0.0 {
            let thr = cfg.min_p * self.probs[0];
            let mut keep = 1;
            while keep < n && self.probs[keep] >= thr {
                keep += 1;
            }
            n = keep;
        }
        // renormalise the kept set
        let s: f32 = self.probs[..n].iter().sum();
        for p in &mut self.probs[..n] {
            *p /= s;
        }
        self.n = n;
        self.kept()
    }

    /// Draw from the kept set of the last `process`.
    pub fn draw(&mut self) -> u32 {
        assert!(self.n > 0, "sampler: draw before process");
        let u = self.rng.f32();
        let mut cum = 0f32;
        for i in 0..self.n {
            cum += self.probs[i];
            if u < cum {
                return self.idx[i];
            }
        }
        self.idx[self.n - 1]
    }

    /// `process` then `draw`.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.process(logits);
        self.draw()
    }

    /// Write the kept set of the last `process` as a dense distribution over the vocabulary (zeros elsewhere).
    pub fn write_dense(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.vocab);
        out.fill(0.0);
        for i in 0..self.n {
            out[self.idx[i] as usize] = self.probs[i];
        }
    }

    /// The speculative acceptance step for one position (`docs/spec.md`): `target_logits` are the main model's
    /// logits at the position, `q` the dense draft distribution the draft token `d` was drawn from. Accepts `d`
    /// with probability `min(1, p(d) / q(d))`; otherwise resamples from `norm(max(0, p - q))`. Returns
    /// `(accepted, token)`: the token is `d` when accepted, the resampled token when not. Greedy: accepted
    /// iff `d` is the argmax, else the argmax.
    pub fn accept_or_resample(&mut self, target_logits: &[f32], q: &[f32], d: u32) -> (bool, u32) {
        self.process(target_logits);
        if self.cfg.is_greedy() {
            let a = self.idx[0];
            return (a == d, a);
        }
        let pd = (0..self.n).find(|&i| self.idx[i] == d).map_or(0.0, |i| self.probs[i]);
        let qd = q[d as usize];
        let u = self.rng.f32();
        if qd > 0.0 && (pd >= qd || u < pd / qd) {
            return (true, d);
        }
        // residual: max(0, p - q) over the kept set of p, renormalised
        let mut sum = 0f32;
        for i in 0..self.n {
            let r = (self.probs[i] - q[self.idx[i] as usize]).max(0.0);
            self.probs[i] = r;
            sum += r;
        }
        if sum <= 0.0 {
            // p sits entirely under q: only possible when p == q on the kept set, in which case the draw above
            // accepted; kept as a guard against rounding
            self.process(target_logits);
            return (false, self.draw());
        }
        for p in &mut self.probs[..self.n] {
            *p /= sum;
        }
        (false, self.draw())
    }
}

/// A minimal JSON reader for `generation_config.json` (flat objects of numbers, booleans, strings, arrays).
pub mod json {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    struct P<'a> {
        s: &'a [u8],
        i: usize,
    }

    impl P<'_> {
        fn ws(&mut self) {
            while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
                self.i += 1;
            }
        }
        fn expect(&mut self, c: u8) -> Result<(), String> {
            self.ws();
            if self.s.get(self.i) == Some(&c) {
                self.i += 1;
                Ok(())
            } else {
                Err(format!("expected {:?} at byte {}", c as char, self.i))
            }
        }
        fn value(&mut self) -> Result<Json, String> {
            self.ws();
            match self.s.get(self.i).copied() {
                None => Err("unexpected end".into()),
                Some(b'{') => {
                    self.i += 1;
                    let mut out = Vec::new();
                    self.ws();
                    if self.s.get(self.i) == Some(&b'}') {
                        self.i += 1;
                        return Ok(Json::Obj(out));
                    }
                    loop {
                        self.ws();
                        let k = self.string()?;
                        self.expect(b':')?;
                        let v = self.value()?;
                        out.push((k, v));
                        self.ws();
                        match self.s.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                return Ok(Json::Obj(out));
                            }
                            _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                        }
                    }
                }
                Some(b'[') => {
                    self.i += 1;
                    let mut out = Vec::new();
                    self.ws();
                    if self.s.get(self.i) == Some(&b']') {
                        self.i += 1;
                        return Ok(Json::Arr(out));
                    }
                    loop {
                        out.push(self.value()?);
                        self.ws();
                        match self.s.get(self.i) {
                            Some(b',') => self.i += 1,
                            Some(b']') => {
                                self.i += 1;
                                return Ok(Json::Arr(out));
                            }
                            _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                        }
                    }
                }
                Some(b'"') => Ok(Json::Str(self.string()?)),
                Some(b't') if self.s[self.i..].starts_with(b"true") => {
                    self.i += 4;
                    Ok(Json::Bool(true))
                }
                Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                    self.i += 5;
                    Ok(Json::Bool(false))
                }
                Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                    self.i += 4;
                    Ok(Json::Null)
                }
                Some(_) => {
                    let start = self.i;
                    while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                        self.i += 1;
                    }
                    let t = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
                    t.parse::<f64>().map(Json::Num).map_err(|_| format!("bad number {t:?} at byte {start}"))
                }
            }
        }
        fn string(&mut self) -> Result<String, String> {
            self.expect(b'"')?;
            let mut out = String::new();
            loop {
                let c = *self.s.get(self.i).ok_or("unterminated string")?;
                self.i += 1;
                match c {
                    b'"' => return Ok(out),
                    b'\\' => {
                        let e = *self.s.get(self.i).ok_or("bad escape")?;
                        self.i += 1;
                        match e {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'n' => out.push('\n'),
                            b't' => out.push('\t'),
                            b'r' => out.push('\r'),
                            b'b' => out.push('\u{8}'),
                            b'f' => out.push('\u{c}'),
                            b'u' => {
                                let h = std::str::from_utf8(self.s.get(self.i..self.i + 4).ok_or("bad \\u")?).map_err(|e| e.to_string())?;
                                self.i += 4;
                                let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            _ => return Err("bad escape".into()),
                        }
                    }
                    _ => {
                        // copy the UTF-8 sequence starting at c
                        let start = self.i - 1;
                        let len = if c < 0x80 {
                            1
                        } else if c >> 5 == 0b110 {
                            2
                        } else if c >> 4 == 0b1110 {
                            3
                        } else {
                            4
                        };
                        let t = std::str::from_utf8(self.s.get(start..start + len).ok_or("bad utf-8")?).map_err(|e| e.to_string())?;
                        out.push_str(t);
                        self.i = start + len;
                    }
                }
            }
        }
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = P { s: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_defaults_are_read() {
        let text = r#"{ "bos_token_id": 248044, "do_sample": true, "eos_token_id": [248046, 248044], "pad_token_id": 248044, "temperature": 1.0, "top_k": 20, "top_p": 0.95 }"#;
        let (c, given) = SamplerConfig::from_generation_config_json(text).unwrap();
        assert_eq!(c, SamplerConfig { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 0 });
        assert_eq!(given, vec!["temperature=1", "top_k=20", "top_p=0.95", "do_sample=true"]);
        let (g, _) = SamplerConfig::from_generation_config_json(r#"{"do_sample": false, "temperature": 0.7}"#).unwrap();
        assert!(g.is_greedy());
        assert!(SamplerConfig::from_generation_config_json(r#"{"repetition_penalty": 1.1}"#).is_err());
    }

    #[test]
    fn top_k_top_p_min_p_keep_the_expected_sets() {
        let logits = [1.0f32, 2.0, 3.0, 0.0, -1.0, 4.0];
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 3, top_p: 1.0, min_p: 0.0, seed: 1 }, 6);
        let (ids, ps) = s.process(&logits);
        assert_eq!(ids, &[5, 2, 1]);
        assert!((ps.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // softmax over all six: 0.633, 0.233, 0.086, 0.032, 0.012, 0.004 (ids 5, 2, 1, 0, 3, 4)
        // top-p 0.5: the first candidate alone has 0.633 > 0.5 but it is the crossing token
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.5, min_p: 0.0, seed: 1 }, 6);
        assert_eq!(s.process(&logits).0, &[5]);
        // top-p 0.9: 0.633 + 0.233 = 0.866 < 0.9, the third crosses (0.951) and stays
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.9, min_p: 0.0, seed: 1 }, 6);
        assert_eq!(s.process(&logits).0, &[5, 2, 1]);
        // top-p 0.85: 0.633 < 0.85, then 0.866 crosses at the second
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 0.85, min_p: 0.0, seed: 1 }, 6);
        assert_eq!(s.process(&logits).0, &[5, 2]);
        // min-p 0.2: keep p >= 0.2 x 0.633 = 0.127: [0.633, 0.233] stay, 0.086 goes
        let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.2, seed: 1 }, 6);
        assert_eq!(s.process(&logits).0, &[5, 2]);
        // greedy
        let mut s = Sampler::new(SamplerConfig::greedy(), 6);
        assert_eq!(s.process(&logits).0, &[5]);
        assert_eq!(s.sample(&logits), 5);
    }

    #[test]
    fn seeded_draws_are_reproducible() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 7919) % 101) as f32 / 10.0).collect();
        let mut a = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 42 }, 1000);
        let mut b = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 42 }, 1000);
        let mut c = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 43 }, 1000);
        let xa: Vec<u32> = (0..200).map(|_| a.sample(&logits)).collect();
        let xb: Vec<u32> = (0..200).map(|_| b.sample(&logits)).collect();
        let xc: Vec<u32> = (0..200).map(|_| c.sample(&logits)).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
    }
}
