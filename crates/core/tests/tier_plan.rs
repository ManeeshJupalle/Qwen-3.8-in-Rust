//! Phase 4.1: the memory plan against hand-computed values.
//!
//! The spans, tensor sizes and dimensions below are copied from `docs/data/gguf_layout.txt` and
//! `tests/fixtures/config.json`, and the arithmetic is redone here by hand (a second, independent
//! implementation of the same rules: the always-resident set, a ring of `n_slots` arenas sized for the
//! largest streamed layer, then whole layers from 0 while they fit). `PlanInput::new` reads the real GGUF
//! header (no tensor data) so the engine's own numbers are compared with the hand ones line by line, and
//! `MemoryPlan::compute` is checked at 6 / 8 / 12 / 16 / 32 GiB. A 3 GiB budget (and 2 GiB) must be refused
//! with a message naming the shortfall. The second test checks the scratch formula against the live
//! `State` on the tiny model, so the plan's scratch line is a measurement, not an estimate.

mod common;

use aqueduct_core::model::Model;
use aqueduct_core::os::arena_bytes;
use aqueduct_core::tier::{MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{Gguf, ModelConfig};

const SECTOR: u64 = 4096;
const GIB: u64 = 1 << 30;

/// Layer spans 0..64 from docs/data/gguf_layout.txt (span = max_end - min_offset).
const SPANS: [u64; 64] = [
    269681536, 269681536, 269681536, 254568448, 269681536, 269681536, 269681536, 254568448, 230974336, 230974336, 269681536, 214018048, 230974336, 253952896, 246702976, 214018048, 253952896, 230974336, 246702976, 254568448, 230974336, 230974336, 269681536, 214018048, 230974336, 253952896, 246702976, 214018048,
    253952896, 230974336, 246702976, 254568448, 230974336, 230974336, 269681536, 214018048, 230974336, 253952896, 246702976, 214018048, 253952896, 230974336, 246702976, 254568448, 230974336, 230974336, 269681536, 214018048, 230974336, 253952896, 246702976, 214018048, 253952896, 230974336, 246702976, 254568448, 269681536, 269681536,
    269681536, 254568448, 269681536, 269681536, 269681536, 254568448,
];
/// blk.64 span; the non-layer region (output.weight at 10995296 .. token_embd end 1769121376).
const MTP_SPAN: u64 = 17772537440 - 17533554272;
const NON_LAYER_SPAN: u64 = 1769121376 - 10995296;

/// `ActBuf::with_capacity(n)`: a Q8_0 row (n codes + per-32: f16 d, f32 d32, f32 dxs, 8 x i32 off6) and a
/// Q8_K row (n codes + per-256: f32 d, 16 x i16 bsums, 8 x i16 q8s).
fn act(n: u64) -> u64 {
    (n + n / 32 * (2 + 4 + 4 + 32)) + (n + n / 256 * (4 + 32 + 16))
}

/// The scratch line by hand: every buffer `State` / `Scratch` / the mixers allocate, plus the logits.
fn hand_scratch(max_pos: u64) -> u64 {
    let (h, inter, vocab) = (5120u64, 17408u64, 248320u64);
    let (cd, nv, dk, dv) = (10240u64, 48u64, 128u64, 128u64);
    let (nh, nkv, hd, rd) = (24u64, 4u64, 256u64, 64u64); // attention.head_count 24 (attn_q rows 12288 = 2 x 24 x 256)
    let vd = nv * dv;
    let mut b = 3 * h * 4 + act(h); // State: h0, h1, lm_normed, lm_act
    b += 3 * h * 4 + act(h); // Scratch: normed, mixed, resid, act
    b += 3 * inter * 4 + act(inter); // MlpScratch
    b += (cd + vd + nv + nv + cd + vd + nv * dk + nv * dk + vd) * 4 + act(vd); // DeltaScratch
    b += (2 * nh * hd + nkv * hd + nkv * hd + rd + rd + nh * hd + nh * hd + hd + nh * hd) * 4 + act(nh * hd); // AttnScratch
    b += 2 * nh * max_pos * 4; // scores, probs
    b += vocab * 4; // logits
    b
}

fn hand_resident(max_pos: u64) -> u64 {
    let small = 48 * (20480 + 20480 + 192 + 192 + 163840 + 512) + 16 * (20480 + 20480 + 1024 + 1024);
    let dn_state = 48 * (48 * 128 * 128 + 10240 * 3) * 4;
    let kv_per_pos = 16 * 2 * 4 * 256 * 4;
    arena_bytes(NON_LAYER_SPAN, SECTOR) + arena_bytes(MTP_SPAN, SECTOR) + small + dn_state + kv_per_pos * max_pos + hand_scratch(max_pos) + (256 << 20)
}

/// (pinned, ring bytes, total) by hand, or None when nothing fits.
fn hand_plan(budget: u64, max_pos: u64, n_slots: u64) -> Option<(u32, u64, u64)> {
    let res = hand_resident(max_pos);
    let mut prefix = vec![0u64];
    for s in SPANS {
        prefix.push(prefix.last().unwrap() + arena_bytes(s, SECTOR));
    }
    if budget >= res + prefix[64] {
        return Some((64, 0, res + prefix[64]));
    }
    for k in (0..64usize).rev() {
        let largest = SPANS[k..].iter().copied().max().unwrap();
        let ring = n_slots * arena_bytes(largest, SECTOR);
        if res + ring + prefix[k] <= budget {
            return Some((k as u32, ring, res + ring + prefix[k]));
        }
    }
    None
}

#[test]
fn plan_matches_hand_computation_at_five_budgets_and_refuses_three_gib() {
    let Some(gguf) = common::gguf_if_present() else { return };
    let g = Gguf::open(gguf).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let input = PlanInput::new(&g, &cfg);

    // the inputs the engine derived, against the layout file
    assert_eq!(input.n_layer, 64);
    for (i, s) in SPANS.iter().enumerate() {
        assert_eq!(input.layer_bytes(i), *s, "layer {i} span");
    }
    assert_eq!(input.mtp, vec![(64, MTP_SPAN, 17533554272)]);
    assert_eq!(input.non_layer_region, Some((10995296, 1769121376)));
    assert_eq!(input.non_layer_arena_bytes(SECTOR), 1_758_130_176);
    assert_eq!(arena_bytes(MTP_SPAN, SECTOR), 238_989_312);
    assert_eq!(input.layer_small.iter().sum::<u64>(), 10_561_536);
    assert_eq!(input.layer_small[0], 205_696, "DeltaNet layer: two norms, ssm_a, dt_bias, conv1d, ssm_norm");
    assert_eq!(input.layer_small[3], 43_008, "attention layer: two norms, q_norm, k_norm");
    assert_eq!(input.deltanet_state_bytes(), 156_893_184);
    assert_eq!(input.kv_bytes_per_pos(), 131_072);
    assert_eq!(input.scratch_bytes(4096), hand_scratch(4096));
    assert_eq!(input.scratch_bytes(4096), 2_589_680);
    assert_eq!(input.scratch_bytes(72), 1_817_072);
    assert_eq!(input.n_head, 24);

    // the plan at the ladder budgets, for the `plan` default of 4096 positions and the ladder's 72
    let expect_4096: [(u64, u32, u64); 5] = [(6, 11, 12_905_576_576), (8, 20, 10_759_711_616), (12, 38, 6_451_640_960), (16, 56, 2_127_226_112), (32, 64, 0)];
    let expect_72: [(u64, u32, u64); 5] = [(6, 13, 12_460_584_192), (8, 22, 10_297_762_944), (12, 40, 5_990_919_936), (16, 58, 1_587_863_040), (32, 64, 0)];
    for (max_pos, table) in [(4096u64, expect_4096), (72u64, expect_72)] {
        let params = PlanParams::new(None, max_pos, 2, SECTOR);
        let plan = MemoryPlan::compute(&input, &params);
        assert_eq!(plan.resident_bytes, hand_resident(max_pos), "resident set at max_pos {max_pos}");
        for (gib, pinned, streamed) in table {
            let budget = gib * GIB;
            let params = PlanParams::new(Some(budget), max_pos, 2, SECTOR);
            let plan = MemoryPlan::compute(&input, &params);
            let (hp, hring, htotal) = hand_plan(budget, max_pos, 2).expect("feasible");
            assert_eq!(plan.pinned, pinned, "{gib} GiB / {max_pos}: pinned layers");
            assert_eq!(plan.pinned, hp);
            assert_eq!(plan.ring_bytes, hring, "{gib} GiB: ring");
            assert_eq!(plan.total, htotal, "{gib} GiB: total");
            assert_eq!(plan.streamed_bytes_per_pass, streamed, "{gib} GiB: bytes per token from disk");
            assert_eq!(plan.streamed_bytes_per_pass, SPANS[pinned as usize..].iter().sum::<u64>());
            assert!(plan.total <= budget && plan.fits(), "{gib} GiB: total {} over budget", plan.total);
            plan.check().expect("fits");
            if pinned < 64 {
                assert_eq!((plan.n_slots, plan.slot_bytes), (2, 269_688_832), "ring for the 269,681,536-byte layers");
                assert_eq!(plan.largest_streamed.map(|(_, b)| b), Some(269_681_536));
            } else {
                assert_eq!((plan.n_slots, plan.ring_bytes), (0, 0), "everything resident: no ring");
            }
            println!("{gib} GiB / max_pos {max_pos}: pinned {} streamed {} ring {} total {} headroom {}", plan.pinned, plan.streamed(), plan.ring_bytes, plan.total, plan.headroom());
        }
    }
    // the printed table adds up to the total
    let plan = MemoryPlan::compute(&input, &PlanParams::new(Some(8 * GIB), 4096, 2, SECTOR));
    let table = plan.table();
    assert!(table.contains("tier 1 (pinned): layers 0..20 (20 layers)"), "{table}");
    assert!(table.contains("tier 2 (streamed): layers 20..64 (44 layers)"), "{table}");
    println!("{table}");

    // an impossible budget: refused, naming the shortfall
    let minimum = hand_resident(4096) + 2 * arena_bytes(269_681_536, SECTOR);
    assert_eq!(minimum, 3_511_847_920);
    assert_eq!(hand_resident(4096), 2_972_470_256);
    for gib in [2u64, 3] {
        let plan = MemoryPlan::compute(&input, &PlanParams::new(Some(gib * GIB), 4096, 2, SECTOR));
        assert!(!plan.fits());
        assert!(hand_plan(gib * GIB, 4096, 2).is_none());
        let e = plan.check().unwrap_err().to_string();
        let shortfall = minimum - gib * GIB;
        assert!(e.contains(&format!("short by {shortfall} bytes")), "{e}");
        assert!(e.contains(&format!("needs {minimum} bytes")), "{e}");
        println!("{gib} GiB: {e}");
    }
    // three slots cost one more arena and pin fewer layers at 6 GiB
    let plan3 = MemoryPlan::compute(&input, &PlanParams::new(Some(6 * GIB), 4096, 3, SECTOR));
    assert_eq!(plan3.ring_bytes, 3 * 269_688_832);
    assert_eq!(plan3.pinned, hand_plan(6 * GIB, 4096, 3).unwrap().0);
    assert_eq!(plan3.pinned, 10);

    // Phase 5: the speculative lines. The snapshot is one copy of the DeltaNet state, the MTP cache one
    // attention layer's KV per position; the scratch line is measured against the live SpecState below.
    assert_eq!(input.spec_snapshot_bytes(), 156_893_184);
    assert_eq!(input.spec_mtp_kv_bytes(4096), 8192 * 4096);
    let saved_k4 = 48 * 5 * (10240 + 96) * 4;
    assert_eq!(saved_k4, 9_922_560);
    assert!(input.spec_scratch_bytes(4, 4096) > saved_k4);
    for k in [1u64, 4] {
        let mut ps = PlanParams::new(Some(6 * GIB), 4096, 2, SECTOR);
        ps.spec_k = k;
        let plan = MemoryPlan::compute(&input, &ps);
        let base = MemoryPlan::compute(&input, &PlanParams::new(Some(6 * GIB), 4096, 2, SECTOR));
        assert_eq!(plan.resident_bytes, base.resident_bytes + input.spec_bytes(k, 4096), "spec k={k}: resident set grows by the spec lines");
        assert!(plan.table().contains(&format!("spec k={k}: DeltaNet state snapshot")), "{}", plan.table());
        println!("6 GiB with --spec {k}: +{} bytes resident ({:.1} MiB), pinned {} (vs {} without)", input.spec_bytes(k, 4096), input.spec_bytes(k, 4096) as f64 / 1048576.0, plan.pinned, base.pinned);
    }
}

#[test]
fn state_bytes_match_the_plan_formula_on_the_tiny_model() {
    let path = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(path.exists(), "tiny GGUF missing at {}", path.display());
    let g = Gguf::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let input = PlanInput::new(&g, &cfg);
    let model = Model::load(&path, 2).expect("load tiny");
    for max_pos in [40u64, 200] {
        let mut state = model.new_state();
        state.reserve(max_pos as usize);
        let planned = input.scratch_bytes(max_pos) - input.vocab * 4 + input.deltanet_state_bytes() + input.kv_bytes_per_pos() * max_pos;
        assert_eq!(state.bytes() as u64, planned, "max_pos {max_pos}: live State bytes vs the plan formula");
        println!("max_pos {max_pos}: State holds {} bytes, plan {planned}", state.bytes());
        // Phase 5: with the speculative buffers for k drafts
        for k in [1u64, 2, 4] {
            let mut s = model.new_state_spec(k as usize).expect("spec state");
            s.reserve(max_pos as usize);
            let planned_spec = planned + input.spec_bytes(k, max_pos);
            assert_eq!(s.bytes() as u64, planned_spec, "max_pos {max_pos}, spec k={k}: live State bytes vs the plan formula");
            println!("max_pos {max_pos}, spec k={k}: State holds {} bytes, plan {planned_spec} (spec part {})", s.bytes(), input.spec_bytes(k, max_pos));
        }
    }
    // the model's own plan (no budget) says everything is resident and no ring exists
    assert_eq!((model.plan.pinned, model.plan.n_slots), (cfg.n_layer, 0));
    assert!(model.stream_stats().is_none());
}
