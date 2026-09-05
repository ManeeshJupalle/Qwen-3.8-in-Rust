//! Phase 5.3: the speculative acceptance rule is distribution-preserving. 20,000 tokens are drawn from a fixed
//! 3-way target `p` through the draft-accept-resample rule with a fixed 3-way draft `q` (and with a point-mass
//! draft, the greedy-draft case), and the histogram is compared with `p`'s expected counts by a chi-square
//! statistic with 2 degrees of freedom. Tolerance: 13.82, the 99.9 % point of chi-square(2). The plain sampler
//! is held to the same tolerance on the same `p`, and seeded runs must reproduce.

use aqueduct_core::sample::{Rng, Sampler, SamplerConfig};

const N: usize = 20_000;
const CHI2_TOL: f64 = 13.82;

fn chi2(counts: &[u64], p: &[f32]) -> f64 {
    counts.iter().zip(p).map(|(&c, &pi)| {
        let e = N as f64 * pi as f64;
        (c as f64 - e).powi(2) / e
    }).sum()
}

fn draw_from(rng: &mut Rng, q: &[f32]) -> u32 {
    let u = rng.f32();
    let mut cum = 0f32;
    for (i, &qi) in q.iter().enumerate() {
        cum += qi;
        if u < cum {
            return i as u32;
        }
    }
    (q.len() - 1) as u32
}

fn histogram(p: &[f32], q: &[f32], seed: u64) -> (Vec<u64>, u64) {
    let logits: Vec<f32> = p.iter().map(|x| x.ln()).collect();
    let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed };
    let mut s = Sampler::new(cfg, p.len());
    let mut rng = Rng::seed(seed ^ 0xA5A5);
    let mut counts = vec![0u64; p.len()];
    let mut accepted = 0u64;
    for _ in 0..N {
        let d = draw_from(&mut rng, q);
        let (ok, tok) = s.accept_or_resample(&logits, q, d);
        counts[tok as usize] += 1;
        accepted += ok as u64;
    }
    (counts, accepted)
}

#[test]
fn speculative_sampling_preserves_the_target_distribution() {
    let p = [0.5f32, 0.3, 0.2];
    for (what, q) in [("q != p", [0.2f32, 0.5, 0.3]), ("q == p", [0.5f32, 0.3, 0.2]), ("point-mass draft", [0.0f32, 1.0, 0.0])] {
        let (counts, accepted) = histogram(&p, &q, 7);
        let x = chi2(&counts, &p);
        println!("{what}: counts {counts:?} (expected {:?}), accepted {accepted}/{N} = {:.3}, chi2 {x:.2} vs tolerance {CHI2_TOL}", p.map(|v| (v * N as f32) as u64), accepted as f64 / N as f64);
        assert!(x < CHI2_TOL, "{what}: chi2 {x} exceeds {CHI2_TOL}");
        // expected acceptance: sum_x q(x) min(1, p(x)/q(x)) = sum_x min(p, q)
        let exp: f32 = p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum();
        let got = accepted as f64 / N as f64;
        assert!((got - exp as f64).abs() < 0.02, "{what}: acceptance {got:.3} vs expected {exp:.3}");
    }
    // the plain sampler on the same p
    let logits: Vec<f32> = p.iter().map(|x| x.ln()).collect();
    let mut s = Sampler::new(SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 7 }, 3);
    let mut counts = vec![0u64; 3];
    for _ in 0..N {
        counts[s.sample(&logits) as usize] += 1;
    }
    let x = chi2(&counts, &p);
    println!("plain: counts {counts:?}, chi2 {x:.2}");
    assert!(x < CHI2_TOL);
    // reproducible per seed
    let (a, _) = histogram(&p, &[0.2, 0.5, 0.3], 11);
    let (b, _) = histogram(&p, &[0.2, 0.5, 0.3], 11);
    assert_eq!(a, b);
}
