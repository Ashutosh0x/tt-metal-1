# Softmax collapse in the fused cross-entropy

Training in tt-train (NanoLlama3 via `sources/examples/nano_gpt`, mnist, lora_llama,
the GRPO log-prob path) computes its loss with `ops::cross_entropy_loss`, which calls
the fused `metal::cross_entropy_fw` / `metal::cross_entropy_bw` device operations.
This branch brings the Section 2 fix of
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642), already in
the fused `log_softmax` ([LOG_SOFTMAX_FUSED.md](LOG_SOFTMAX_FUSED.md)), to those two
operations.

## Did the old kernels collapse? Yes, in both directions

Notation per row: `k` the first maximum, `m = x[k]`, `T = sum_{j != k} exp(x[j] - m)`.
For a target `t = k` the exact loss is `log1p(T) ~ T` and the exact target gradient is
`-s T / (1 + T)` (`s` = scaler, `1/N` for MEAN). Collapse means both round to exactly 0
while `T` is still representable: the model stops being pushed on tokens it already
predicts confidently, which is most tokens late in training.

**Forward** (`cross_entropy_fw_kernel.cpp`, removed here). Per SFPU lane it summed
`exp(x - m)` over the whole row *including* the maximum's `e^0 = 1`, in FP32 DST. The
32 lane sums were row-reduced with an FPU matmul against a ones tile, then `log` was
taken, and the loss was `-x_t + m + log(sum)`.

- `log(1 + T)` of an FP32 `1 + T` is exactly 0 once `T < 2^-24`: one competitor at a gap
  of 17 or more, or a 32000-wide flat tail at a gap of about 27 or more.
- Earlier than that, the matmul reads its FP32 operand through the FPU source registers,
  which keep TF32 (10 explicit mantissa bits), and the lane holding the maximum carries
  `1 + t_lane`. Small tails are therefore resolved only to about `2^-10`. On hardware the
  confident losses came out as exactly 0 or exactly `2^-9` (0.00195): `2^-9` for a true
  3.4e-4 at gap 8, and for a true mean loss of 8.4e-6 (233 times too high), 0 from gap 9.
- The BH `sdpa_exp_tile` is a BF16-precision exponential, so even non-collapsed tails
  carry BF16-level relative error.

**Backward** (`cross_entropy_bw_kernel.cpp`, removed here). It computed
`p_j * s = exp(x_j - m) * recip(sum) * s`, packed the tile to BF16, and the writer then
subtracted `s` at the target *after* the BF16 rounding: `dx_t = bf16(p_t s) - s`. With
`p_t = 1 / (1 + T)`, `bf16(p_t s)` is exactly `s` once `T < 2^-9`, so the target
gradient is exactly 0 from a single-competitor gap of about 6.3 (a 32000-wide tail at a
gap of about 17). The non-target gradients `s p_j` are fine.

**Proof** (`tests/ops/cross_entropy_collapse_test.cpp`, committed before the fix, run
against the old kernels on Blackhole p100a and Wormhole n150): all four tests fail on both,
while the 17 existing cross-entropy tests pass (their `allclose(rtol 3e-2, atol 1e-2)`
cannot see a loss or gradient of 1e-4 becoming 0).

| Case (rows: target = first maximum, gaps 8-24) | BH mismatches | WH mismatches |
| --- | ---: | ---: |
| `[1,1,32,32000]` forward / backward | 46 / 54 | 54 / 54 |
| `[2,1,35,1000]` forward / backward | 110 / 66 | 124 / 66 |
| `[1,1,64,35]` forward / backward | 106 / 122 | 118 / 122 |
| `[4,1,64,4096]` forward / backward | 370 / 442 | 428 / 442 |
| `[2,1,256,32000]` mixed rows, forward / backward | 106 / 70 | 124 / 64 |
| `[1,1,33,1025]` mixed rows, forward / backward | 8 / 5 | 10 / 5 |
| CE vs corrected log_softmax, `[1,1,32,32000]` / `[2,1,35,1000]` | 55 / 133 | 55 / 128 |
| `ops::cross_entropy_loss` MEAN autograd, `[2,1,32,1024]` | 128 + loss | 128 + loss |

Mismatch counts include both the BF16-tolerance check against float64 and the explicit
"loss > 0" / "target gradient < 0" checks. Every confident target gradient from gap 8 to
24 was exactly 0 (for example `-1.05e-5` expected, `0` actual), and so was every
confident loss from gap 9 up.

## Fix

The fused cross-entropy now runs the fused log_softmax's device program
(`metal/ops/log_softmax/row_softmax_program.{hpp,cpp}`) with a `CROSS_ENTROPY` kernel
define; both device operations keep their API, validation, program-cache key (shape,
dtype, scaler) and preallocated-output support.

- **Statistics** (shared with log_softmax, unchanged): one online pass per row keeps,
  per SFPU lane, the running maximum, the tile column of its first maximum and the tail
  of every other element, `t' = (t + 1) exp(m - x)` on a new maximum. The lane combine
  selects exactly one first maximum `k` and never adds its `e^0`; `l = log1p(T)` in FP32.
  All FP32 loads use unpack-to-destination, so nothing passes through TF32.
- **Forward**: the reader fetches each row's target logit with one aligned 64-byte read
  (the row's 32 targets per tile row), so the logits are read once instead of twice.
  `loss = -((x_t - m) - l)` is computed exactly as the log_softmax output at `t`; for
  `t = k` it is `l` itself.
- **Backward**: one more pass, `dx_j = exp(x_j - m) * q` with `q = s exp(-l)` (one
  exponential per element, BF16 out). The writer substitutes the target element with
  `s * expm1((x_t - m) - l) = s (p_t - 1)`. For `t = k` that is `-s T / (1 + T)`, the
  sum of the other gradients (Eq. 7) in closed form; for any other target it equals
  `s p_t - s` without the BF16 double rounding. No branch is needed: `x_k - m = 0`.
- Out-of-range targets keep the old convention (zero target logit, no `-s` term), and
  vocabulary padding still gets zero gradient (`-inf` in the reader).
- Few-row shapes split a row across up to 32 cores with the log_softmax's DRAM
  exchange; the scratch buffer is owned by the cached program.
- The logits are read once forward (was twice) and twice backward (was three times).
  The backward still recomputes the statistics (the public API passes no state).
- Separate commit: `metal::cross_entropy_bw` used to multiply the full `[N, 1, H, V]`
  result by the upstream gradient with a second `ttnn::multiply`. The prim now takes the
  upstream gradient, one value (`[1,1,1,1]`, MEAN) or one per row (`[N,1,H,1]`, NONE),
  and the kernel folds it into each row's `s`; other shapes keep the multiply.

The accurate FP32 exponential costs more per element than the old Blackhole
BF16-precision `sdpa_exp_tile`, which is why the fix alone is slower at NanoLlama3
geometry; folding in the upstream gradient more than pays for it.

## Results

Variants: *old* = the cross-entropy kernels before this change, *fix* = the collapse fix
alone, *fix + upstream* = this change (the fix with the upstream gradient folded into the
backward). Same clean build per machine, only tt-train rebuilt between variants.

### Correctness

On both Blackhole p100a (IRD bh-33) and Wormhole n150 (IRD wh-14), with fix + upstream:
6/6 `CrossEntropyCollapseTest`, 17/17 existing `CrossEntropyForwardTest.*` /
`CrossEntropyBackwardTest.*`, and 64/64 log_softmax tests (the fused log_softmax suite, which
covers the refactored shared kernels). The fix alone also passed 5/5, 17/17 and 64/64 on
both. No existing tolerance changed; the new tests use the log_softmax tests' relative
tolerances (BF16 `1e-2`, FP32 `3e-5` plus the FP32 accumulation allowance) with no
absolute slack beyond flushed denormals.

### Operation latency and DRAM

`ops::cross_entropy_loss(MEAN)` forward + backward, BF16 logits, median ms (C++
`steady_clock` around eager dispatch with device synchronization,
`tests/benchmark/cross_entropy_benchmark.cpp`). Step peak = incremental DRAM of one
forward + backward.

| Shape | BH old | BH fix | BH fix + upstream | WH old | WH fix | WH fix + upstream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[64,1,256,32000] NanoLlama3** | 24.39 | 26.79 | **20.69** | 64.64 | 66.51 | **56.19** |
| [8,1,256,32000] | 4.05 | 4.97 | 4.20 | 8.39 | 8.49 | 7.21 |
| [1,1,256,32000] | 2.82 | 0.59 | 0.50 | 6.72 | 1.36 | 1.20 |
| [1,1,32,32000] | 2.71 | 0.37 | 0.36 | 6.34 | 0.69 | 0.65 |
| [1,1,32,128256] | 10.34 | 0.79 | 0.74 | 24.59 | 1.40 | 1.30 |
| [1,1,1024,4096] | 0.49 | 0.38 | 0.34 | 1.09 | 0.86 | 0.77 |

NanoLlama3 split, Blackhole: forward 6.85 -> 7.04 -> 7.06 ms, backward 17.59 -> 19.79
-> 13.71 ms. Wormhole: forward 18.45 -> 19.87 -> 19.88, backward 46.03 -> 46.69 -> 36.22.

| DRAM, NanoLlama3 | old | fix | fix + upstream |
| --- | ---: | ---: | ---: |
| Forward peak / retained | 2.23 MB / 2 KB | same | same |
| Forward + backward peak | 2.10 GB | 2.10 GB | **1.05 GB** |

Observations:

- At NanoLlama3 geometry the fix alone is 10% slower on Blackhole (3% on Wormhole):
  the accurate FP32 exponential, twice per element, replaces the Blackhole BF16-precision
  one. With the upstream gradient folded in, the loss is 15% (BH) and 13% (WH) faster
  than before and needs half the DRAM.
- Few rows and wide vocabularies gain 5-14x (Blackhole) and 6-19x (Wormhole) from the
  row split across cores and the single forward pass.
- `[8,1,256,32000]` (64 tile rows on 110 cores, no split, a single wave) stays 4% slower
  on Blackhole; it is the case where the exponential cost is not hidden by either change.

### NanoLlama3 training (Blackhole p100a)

`nano_gpt` with `training_shakespeare_nanollama3.yaml` (batch 64, 6 blocks, d=384,
T=256, V=32000, AdamW 3e-4) on Shakespeare tokenized with the TinyLlama tokenizer
(`tools/dataset_to_tokens.py`), `--track_memory`. Step time is the median over steps 11+.

1000 steps per run, three seeds per version. The old seed-1 and seed-2 runs and the
fix-only runs come from the first session; the rest were rerun on the same machine and build.

| 1000 steps | old | fix + upstream (PR) | change |
| --- | ---: | ---: | ---: |
| Median step time, mean of 3 seeds | 118.55 ms | **114.84 ms** | -3.1% |
| Wall time, steps 2-1000, mean of 3 seeds | 118.3 s | **114.6 s** | -3.1% |
| First-iteration cumulative DRAM peak | 4855 MB | **3939 MB** | -19% |
| Mean loss, steps 901-1000, seed 5489 (default) | 0.862 | 0.799 | -0.063 |
| Mean loss, steps 901-1000, seed 1 | 0.837 | 0.837 | same |
| Mean loss, steps 901-1000, seed 2 | 0.808 | 0.794 | -0.014 |
| Mean loss, steps 901-1000, 3-seed average | 0.836 | 0.810 | -3.1% |

The fix alone runs at 120.9 ms per step with the old peak (4855 MB); its loss curves are
identical to the PR's at every logged step (the logged loss is BF16); folding in the
upstream gradient changes only how the gradient is rounded. Fixed and old curves are
identical for the first 31-51 steps and then drift apart. The PR's loss is lower or equal
on all three seeds, but three 1000-step runs cannot separate that from run-to-run noise;
they do show no regression in training quality. Losses here stay above 0.7, where few
tokens reach the collapse regime; the paper's effect grows as the model becomes confident.

## Reproduction

```bash
./build_metal.sh --build-tt-train
export TT_METAL_HOME="$PWD" TT_METAL_RUNTIME_ROOT="$PWD"
build/tt-train/tests/ttml_tests --gtest_filter='CrossEntropyCollapseTest.*:CrossEntropyForwardTest.*:CrossEntropyBackwardTest.*'
build/tt-train/tests/cross_entropy_benchmark
# Training: tokenized data, then nano_gpt with a copy of the config pointing at it.
python tt-train/tools/dataset_to_tokens.py --text_file data/shakespeare.txt \
    --hf_tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output_file data/tokenized_shakespeare_tinyllama
build/tt-train/sources/examples/nano_gpt/nano_gpt -c <config with data_path and max_steps> --track_memory
```

To measure the old kernels, build the benchmark on the parent revision.

## Remaining work

### Vocabulary-parallel loss (`ops::distributed::vocab_parallel_cross_entropy_loss`)

Done: see [VOCAB_PARALLEL_CE_COLLAPSE.md](VOCAB_PARALLEL_CE_COLLAPSE.md). The old composite collapsed (every
confident loss and target gradient exactly 0 from a gap of 8 on a Wormhole n300 TP2 mesh); it now runs three
vocabulary-parallel kinds of the same row-softmax program with one all-gather of per-shard FP32 records.

### Section 3: teacher/student cancellation (GRPO log-prob ratio)

The GRPO trainer (`sources/ttml/ttml/trainers/grpo_trainer.py`) forms the PPO ratio as
`exp(nlog_old - nlog_new)`, where each per-token negative log-probability comes from a
separate `cross_entropy_loss(..., ReduceType.NONE)` call (`compute_nlog_probs` in the
completers) and is rounded to BF16. The Section 2 fix makes each value accurate to BF16
relative precision, including for confident tokens that used to read exactly 0. Their
difference still cancels: two BF16 values near `L` differ in steps of about `2^-8 L`, so
for a token at `L = 4` the log-ratio is quantized to about 0.016, coarser than a typical
per-update policy change, and a small but real change reads as exactly 0 (ratio 1). Section 3 of the paper fixes this with a loss that sees both logit rows and
forms `log p_t - log q_t = (x_t - m_x - l_x) - (z_t - m_z - l_z)` in FP32 from the two
rows' statistics (with `log1p`/`expm1` for the ratio and KL), rounding only the result.
That would be a fused `log_ratio(new_logits, old_logits, targets)` built on the same
row statistics, plus keeping the old policy's per-row `(m, l, x_t)` in FP32 instead of
its BF16 loss. It is not implemented here.
