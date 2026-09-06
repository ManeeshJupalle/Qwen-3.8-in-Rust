`aqueduct` runs Qwen3.8-27B (bartowski's Q4_K_M GGUF, 17.8 GB) on a CPU-only Windows or Linux machine with
8 to 16 GB of RAM, streaming the layers that do not fit from the drive, with the same greedy tokens at every
memory budget. Start with `aqueduct doctor`; read the README's limitations before downloading the model.

## The ladder (one i7-9750H laptop, NVMe, Windows 11; 3 prompts x 32 greedy tokens)

| budget | plain s/token | `--spec 3` s/token | speed-up | peak RSS plain / spec (GB) | disk GB/token |
|---|---|---|---|---|---|
| **5 GiB** (an 8 GB laptop's free RAM) | 4.305 | 1.894 | 2.27 x | 5.103 / 5.057 | 13.406 |
| **11 GiB** (a 16 GB laptop's free RAM) | 2.819 | 1.581 | 1.78 x | 11.569 / 11.557 | 7.008 |
| 6 GiB | 4.095 | 1.979 | 2.07 x | 6.051 / 6.005 | 12.461 |
| 8 GiB | 3.628 | 1.812 | 2.00 x | 8.218 / 8.172 | 10.298 |
| 12 GiB | 2.690 | 2.050 | 1.31 x | 12.535 / 12.504 | 5.991 |
| 16 GiB (hot) | 1.572 | 1.344 | 1.17 x | 16.946 / 16.860 | 1.588 |
| resident (hot) | 1.170 | 1.178 | 0.99 x | 17.961 / 18.127 | 0 |

Sources: `docs/data/ladder.txt` (5, 11, 16 GiB, resident), `docs/data/ladder_phase5.txt` (6, 8, 12 GiB, with the
Phase 5 verify kernel); the 96 output ids are identical at every row (`tests/fixtures/ladder_expected_ids.json`).
The 16 GiB and resident rows were taken hot; cool they measured 1.155 / 1.129 and 0.830 / 1.057 s/token.

Model: `bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`, 17,772,537,440 bytes, sha256
`e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b` (the only supported file). Needs AVX2.
