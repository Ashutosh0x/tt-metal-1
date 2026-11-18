# Fused corrected log_softmax

`ops::log_softmax` routes supported calls through one device program for the forward
and one for the backward. The corrected composite stays available as
`ops::log_softmax_composite`, and is the fallback for unsupported cases (for example
rank-3 inputs).

## Arithmetic

Following [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642),
Section 2: select exactly one first maximum `k`, omit its exponential from the
tail, and compute `log1p(tail)` in FP32. Forward emits `(x - max) - log1p(tail)`.
The selected element's gradient is reconstructed from the other entries (Eq. 7)
instead of subtracting nearly equal values. FP32 input and statistics use
unpack-to-destination mode; routing them through the FPU copy truncates to TF32.

Section 3's teacher/student cancellation needs a logits-aware loss; a log_softmax
API returning rounded log-probabilities cannot provide it. The fused cross-entropy used
for training now shares this program (`row_softmax_program.{hpp,cpp}`); see
[CROSS_ENTROPY_COLLAPSE.md](CROSS_ENTROPY_COLLAPSE.md).

## Kernel design (2026-11-17)

**Forward: two passes over the logits.**

1. *Online statistics.* Each SFPU lane keeps a running maximum `m`, the tile column
   of its first maximum, and a tail `t` = sum of `exp(x - m)` over every element it
   has seen except that one maximum. A strictly larger `x` becomes the new maximum
   and the old one joins the tail: `t' = (t + 1) * exp(m - x)`; ties fall through to
   `t += 1`, so each lane keeps its first maximum. One exponential, `exp(-|x - m|)`,
   serves both branches. The reader writes `-inf` into tile padding for this pass.
   The 32 lanes are then combined per row: `M = max m`, `K = min{index : m == M}`,
   `T = sum t * exp(m - M) + [lane != K] exp(m - M)`. The selected `e^0 = 1` is never
   added to the tail.
2. *Output.* `y = (x - M) - log1p(T)`, with `M` and `log1p(T)` pre-broadcast once per
   row into full tiles, so each tile costs one custom SFPU operation.

**Backward: two passes.**

1. *Sum of the other upstream gradients.* The reader moves each row's selected
   element `dy_k` into a column vector and zeroes it in the streamed tile, so the
   compute pass is a plain FP32 sum `A = sum_{j != k} dy_j`. `S = A + dy_k`.
2. *Gradient.* `dx = dy - exp(x - M) * q` with `q = S * exp(-log1p(T)) = S * p_k`,
   one exponential per element. The writer substitutes the selected element with
   Eq. 7 in closed form, `dx_k = -sum_{j != k} dx_j = S * q' - A`, where
   `q' = sum_{j != k} p_j = -expm1(-log1p(T))` stays accurate when `p_k` rounds to one.

That is two exponentials per element per forward+backward, the minimum without
storing probabilities. The baseline prototype read logits four times forward and
three times backward, with an exponential in two passes of each.

**Few rows: split a row across cores.** One core handles a block of 32 physical rows.
When fewer than half the cores would have a row, each row's vocabulary is split over
up to 32 cores (at least 16 tiles each, one unit per core). After pass 1 every core
publishes a 512-byte record of its per-row statistics to DRAM and signals the row's
first core; that core releases all of them once every record has arrived. Each core
then reads all records with one barrier and repeats the lane combine, with chunks as
lanes, before its pass 2 on its own columns. Backward exchanges `(A, dy_k)` the same
way. Forward's combine is exactly the lane combine above; partial records add
`32 x 512` bytes of DRAM per split row.

**State** is unchanged: three FP32 column-vector tiles per 32 physical rows (maximum,
selected index, `log1p(tail)`), `[1, 1, physical_rows, 96]`, 384 bytes per row,
6 MiB for NanoLlama3.

Each core pushes stream blocks of 4, 2 or 1 tiles, whichever divides its column
count, so every push stays aligned to the two-block ring. (Splitting rows with a
fixed block made a partial last block straddle the ring end and corrupted a tile.)

## Results

BF16 inputs, last axis, forward + backward median, host time with device
synchronization (host dispatch included). Prototype = the first fused version
(see History).

### Blackhole p100a (IRD bh-36)

| Shape | Original (buggy) ms | Composite ms | Prototype ms | **New ms** | vs prototype |
| --- | ---: | ---: | ---: | ---: | ---: |
| [64, 1, 256, 32000] NanoLlama3 | 45.00 | 412.72 | 59.75 | **22.90** | 2.6x |
| [1, 1, 32, 32000] | 1.060 | 1.680 | 11.438 | **0.291** | 39x |
| [1, 1, 256, 32000] | 1.625 | 7.023 | 11.441 | **0.514** | 22x |
| [1, 1, 32, 128256] | 4.136 | 5.817 | 45.727 | **0.646** | 71x |
| [1, 1, 512, 128256] | 8.665 | 39.678 | 45.759 | **3.393** | 13x |
| [1, 1, 1024, 4096] | 0.747 | 6.407 | 1.514 | **0.315** | 4.8x |
| [1, 1, 128, 1024] | 0.442 | 1.150 | 0.412 | **0.151** | 2.7x |
| [1, 1, 32, 128] | 0.363 | 0.990 | 0.091 | **0.074** | 1.2x |
| [1, 1, 1024, 4096], axis 2 | 0.619 | 80.325 | 1.026 | **0.540** | 1.9x |

NanoLlama3 forward 23.25 -> 10.75 ms, backward 36.67 -> 12.18 ms. The fused
operation is now faster than the original, incorrect arithmetic on every measured
shape. Forward retained DRAM is unchanged (state only); split rows add the partial
records.

### Wormhole n150 (IRD wh-14), regression check

| Shape | Prototype ms | New ms |
| --- | ---: | ---: |
| [64, 1, 256, 32000] | 154.73 | 51.05 |
| [1, 1, 32, 32000] | 18.67 | 0.53 |
| [1, 1, 32, 128256] | 74.75 | 1.09 |
| [1, 1, 1024, 4096] | 2.53 | 0.66 |
| [1, 1, 32, 128] | 0.206 | 0.172 |

No shape regressed. Wormhole has 64 worker cores against Blackhole's 110 here.

### Device profile (Blackhole, math thread)

With `TT_METAL_DEVICE_PROFILER=1`, the compute kernel marks the zones `LSM-FW-STATS`,
`LSM-FW-COMBINE`, `LSM-FW-EMIT`, `LSM-BW-SUM`, `LSM-BW-PIVOT` and `LSM-BW-EMIT`.

- NanoLlama3 forward kernel 11.0 ms per core: statistics 6.8 ms (about 1.9k cycles
  per tile, one accurate FP32 exponential per element), output 3.0 ms, combine
  0.03 ms. Backward is dominated by its gradient pass in the same way.
- Before the compact exchange, `[1, 1, 32, 32000]` spent 120-156 us of a 203 us
  forward kernel exchanging partial statistics (96 serial 4 KiB DRAM reads per core);
  packing them into 512-byte records fetched under one barrier removed most of it.
- The FP32 exponential is the floor. Its range-reduction constants are held in the
  programmable constant registers; the remaining polynomial coefficients still cost
  two SFPLOADIs each per row (about 9 of ~40 instructions).

## Testing

`tt-train/tests/ops/log_softmax_test.cpp` adds 19 tests (filter `*FusedLogSoftmax*`):
wide vocabularies (32000-128256) with the maximum at column 0, the last valid column,
partial tiles and many plausible chunk boundaries; ties across far-apart positions
and chunks (first occurrence in the state index and gradient); online-rescaling
stress (monotone, staircase, alternating, constant, +/-1e4 offsets); confidence gaps
8-60 and underflow gaps 80-150; widths 1-1025 and more tile rows than cores; five
upstream patterns including FP32 upstream with BF16 input; agreement with the
composite; the rank-3 composite fallback; and program-cache alternation across
shapes and dtypes. A float64 oracle on the quantized inputs checks every output and
gradient, the pivot reconstruction, `sum(dx) ~ 0`, and the state contract.

The new tests use two width-aware tolerances that the existing ones do not need:
gradient tolerance relative to `|dy_j| + p_j |S|` (identical to `|dx_j|` for the
one-hot collapse cases), and an FP32 sequential-sum allowance of
`(W/32 + 32) * 2^-24` relative to the summed magnitudes (tail, `S`, other gradients).
The prototype itself exceeded fixed 3e-5 bounds at 32k-128k widths with random
upstream. A collapse still fails these checks.

On both Blackhole and Wormhole, the prototype and the new kernels pass all 64 tests
(45 existing log_softmax tests plus the 19 new ones). No existing tolerance changed.

## Reproduction

```bash
cmake --build build --target ttml_tests log_softmax_benchmark
export TT_METAL_HOME="$PWD" TT_METAL_RUNTIME_ROOT="$PWD"
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 build/tt-train/tests/ttml_tests \
  --gtest_filter='UnaryOpsTest.*:PrecisionAndAxes/LogSoftmaxTest.*:TrivialTnnFixedTest.TestMax*:TrivialTnnFixedTest.TestStableSoftmax*:*FusedLogSoftmax*'
build/tt-train/tests/log_softmax_benchmark
```

The benchmark prints forward, backward and forward + backward tables (median µs per
launch, DRAM peak of one step) for the original composite, the corrected composite
and the fused op.

## History: 2026-10-30 prototype

The first fused version read logits four times forward
(maximum, first index, tail, output) and three times backward (upstream sum,
other-gradient sum, output), one 32-row block per core. On a Blackhole P150a it ran
NanoLlama3 in 55.79 ms (original 38.81, composite 377.72) and was 7-10x slower than
the composite on few-row wide shapes.
