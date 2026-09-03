# ggml block layouts implemented in `crates/core/src/quant.rs`

Every ggml type present in the primary GGUF (`Qwen3.8-27B-Q4_K_M.gguf`, bartowski), with the byte layout of
one block as written in the file and the exact arithmetic used to turn it into f32. All multi-byte fields are
little-endian. `f16` is IEEE binary16 and is converted to f32 exactly (`half` crate). The scalar loops follow
`dequantize_row_*` in ggml's `ggml-quants.c`; gguf-py's numpy implementation (the fixture spec) uses the same
operation order, and `crates/core/tests/dequant.rs` confirms the Rust output is bit-identical to gguf-py
(`tests/fixtures/dequant/`, emitted by `tools/emit_dequant_fixtures.py`). No summation is involved in
dequantisation, so no tolerance is needed; the test asserts equality of f32 bit patterns.

Type ids and block sizes (as stored in the tensor infos):

| type | id | elements / block | bytes / block | in this file |
|---|---|---|---|---|
| F32 | 0 | 1 | 4 | 456 tensors (norms, ssm_a, ssm_dt.bias, ssm_conv1d, ssm_alpha/beta) |
| Q4_0 | 2 | 32 | 18 | 8 tensors (the MTP block's matrices) |
| Q8_0 | 8 | 32 | 34 | 24 tensors (half of ssm_out) |
| Q4_K | 12 | 256 | 144 | 249 tensors |
| Q5_K | 13 | 256 | 176 | 32 tensors (attn_k, attn_output) |
| Q6_K | 14 | 256 | 210 | 97 tensors |

Element order inside a tensor: blocks are consecutive along the fastest dimension (`shape[0]` in GGUF order,
the last dimension in numpy order); a row of `ne0` elements is `ne0 / block_size` blocks back to back.

## F32

| offset | size | field |
|---|---|---|
| 0 | 4 | value, f32 |

## Q4_0 (32 elements, 18 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d`, f16 scale |
| 2 | 16 | `qs[16]`, 4-bit codes |

Element `j` (0..16) is the low nibble of `qs[j]`; element `j + 16` is the high nibble.
`y = (q - 8) * d` computed as `((q as i32) - 8) as f32 * d`.

## Q8_0 (32 elements, 34 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d`, f16 scale |
| 2 | 32 | `qs[32]`, int8 codes |

`y[j] = (qs[j] as i8) as f32 * d`.

## Q4_K (256 elements, 144 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d`, f16 super-block scale |
| 2 | 2 | `dmin`, f16 super-block min scale |
| 4 | 12 | `scales[12]`, eight 6-bit scales `sc[0..8]` and eight 6-bit mins `m[0..8]`, packed as below |
| 16 | 128 | `qs[128]`, 4-bit codes |

6-bit packing (`get_scale_min_k4` in ggml). For `j < 4`: `sc[j] = scales[j] & 63`, `m[j] = scales[j+4] & 63`.
For `j >= 4`: `sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)`,
`m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)`. Byte by byte:

| byte | bits 0-5 | bits 6-7 |
|---|---|---|
| 0..3 (`j`) | `sc[j]` | high 2 bits of `sc[j+4]` |
| 4..7 (`j`) | `m[j-4]` | high 2 bits of `m[j]` |
| 8..11 (`j`) | low nibble: low 4 bits of `sc[j-4]`; high nibble: low 4 bits of `m[j-4]` | |

Codes: the 256 elements are 8 sub-blocks of 32. Bytes `qs[32k .. 32k+32]` (k = 0..4) hold sub-block `2k` in
the low nibbles and sub-block `2k+1` in the high nibbles. For sub-block `s` with scale `sc[s]` and min `m[s]`:
`y = (d * sc[s]) * q - (dmin * m[s])`, evaluated as `d1 = d * sc as f32; m1 = dmin * m as f32; y = d1 * q as f32 - m1`.

## Q5_K (256 elements, 176 bytes)

| offset | size | field |
|---|---|---|
| 0 | 2 | `d`, f16 |
| 2 | 2 | `dmin`, f16 |
| 4 | 12 | `scales[12]`, packed exactly as Q4_K |
| 16 | 32 | `qh[32]`, fifth-bit plane |
| 48 | 128 | `qs[128]`, low 4-bit codes, laid out as Q4_K |

High bits: for the pair of sub-blocks `2k, 2k+1` (k = 0..4) and position `l` (0..32), bit `2k` of `qh[l]` is the
fifth bit of the sub-block `2k` element and bit `2k+1` of `qh[l]` is the fifth bit of the sub-block `2k+1`
element (ggml walks `u1 = 1, u2 = 2` and shifts both left by 2 per pair). `q = low4 + (bit ? 16 : 0)`,
then `y = (d * sc) * q - (dmin * m)` as in Q4_K.

## Q6_K (256 elements, 210 bytes)

| offset | size | field |
|---|---|---|
| 0 | 128 | `ql[128]`, low 4-bit codes |
| 128 | 64 | `qh[64]`, high 2-bit codes |
| 192 | 16 | `scales[16]`, int8 sub-block scales (one per 16 elements) |
| 208 | 2 | `d`, f16 super-block scale |

The block is two halves of 128 elements (`h` = 0, 1) using `ql[64h..64h+64]`, `qh[32h..32h+32]`,
`scales[8h..8h+8]`. Inside a half, for `l` in 0..32:

| element | low 4 bits | high 2 bits | scale |
|---|---|---|---|
| `128h + l` | `ql[64h + l] & 0xF` | `(qh[32h + l] >> 0) & 3` | `scales[8h + l/16]` |
| `128h + 32 + l` | `ql[64h + 32 + l] & 0xF` | `(qh[32h + l] >> 2) & 3` | `scales[8h + 2 + l/16]` |
| `128h + 64 + l` | `ql[64h + l] >> 4` | `(qh[32h + l] >> 4) & 3` | `scales[8h + 4 + l/16]` |
| `128h + 96 + l` | `ql[64h + 32 + l] >> 4` | `(qh[32h + l] >> 6) & 3` | `scales[8h + 6 + l/16]` |

`q = (low4 | (high2 << 4)) - 32` (range -32..31), `y = d * sc * q` evaluated left to right as
`(d * (sc as i8) as f32) * q as f32`.

## Not implemented (present in the Unsloth UD file, absent from bartowski's)

IQ4_XS, IQ4_NL, IQ3_S, Q3_K. `GgmlType` knows their ids and block sizes so the index is still exact for
those files; `dequantize` returns `QuantError::Unsupported` for them.

## Fixtures

`tests/fixtures/dequant/manifest.json` lists, per type: the smallest tensor of that type (raw bytes when
<= 8 MiB, otherwise read from the GGUF at test time; first 4096 f32 values; CRC-32 of the full f32 stream;
max|x|), the first 3 blocks of the largest tensor (raw + f32), and the hostile hand-built blocks:

| label | what is hostile about it |
|---|---|
| q4_0 | low and high nibbles are different permutations of 0..15 |
| q8_0 | int8 codes include -128 and 127 |
| q4_k | all 8 scales and all 8 mins are distinct 6-bit values that use both halves of every packed byte (63, 1, 2, 4, 8, 16, 32, 33 / 62, 3, 5, 9, 17, 31, 47, 60) |
| q5_k | same scales, plus a `qh` plane whose bits differ in every position |
| q6_k | 16 distinct int8 scales including 127 and -128, `ql`/`qh` patterns that differ per element |
| f32 | +0, -0, inf, -inf, a subnormal, 3.4e38 |

`tests/dequant.rs::wrong_unpack_is_caught_by_hostile_q4_k` checks that a deliberately wrong unpack
(scale/min swapped) does not reproduce the fixture.
