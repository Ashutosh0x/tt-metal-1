# Softmax collapse in the vocabulary-parallel cross-entropy

Under tensor parallelism (`device_config.enable_tp`), nano_gpt and lora_llama compute their loss with
`ops::distributed::vocab_parallel_cross_entropy_loss`: the LM head emits logits sharded along the
vocabulary, `[B, 1, S, V / tp]` per device, and the loss never gathers the full row. This branch checks
that path for the softmax collapse of
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642) (Section 2) and fixes it with the
same arithmetic as the single-device fused cross-entropy ([CROSS_ENTROPY_COLLAPSE.md](CROSS_ENTROPY_COLLAPSE.md))
and fused log_softmax ([LOG_SOFTMAX_FUSED.md](LOG_SOFTMAX_FUSED.md)).

Notation per row: `k` the first maximum over the whole (all-shard) row, `M = x[k]`,
`T = sum_{j != k} exp(x[j] - M)`. For a target `t = k` the exact loss is `log1p(T) ~ T` and the exact target
gradient is `-s T / (1 + T)` (`s` = 1/N for MEAN, times the upstream gradient).

## Analysis of the old implementation

`losses.cpp` was a composite of TTNN ops and three collectives:

| Step | Operation | Precision |
| --- | --- | --- |
| local max | `ttnn::max(logits, dim 3)` | BF16, exact |
| global max | `all_gather` of the `[B,1,S,1]` local maxima, then `ttnn::max` | BF16, exact |
| local sum | `exp(x - M)` fused into `ttnn::subtract` (binary_ng post-activation), FP32 out; `ttnn::sum` (precise) | FP32 |
| global sum | `all_reduce` of the FP32 local sums | FP32 CCL reduction |
| normalizer | `M + log(global_sum)`, FP32 | FP32 |
| target logit | `select_target_logit` (BF16, zero off-shard), `all_reduce` | BF16, exact |
| loss | `(M + log(sum)) - x_t` in FP32, then BF16 (NONE) or mean (MEAN) | FP32 |
| backward | `bf16(exp(x - M) * inv_N / sum)`, then `subtract_at_target` subtracts `inv_N` | BF16 |

**Where the global maximum is taken.** On every device from the all-gathered local maxima, so all devices
agree on `M` exactly. No device selects *which* element is the maximum: there is no first-maximum index at
all, so nothing can be excluded from the sum.

**Does the dominant `e^0 = 1` swamp the tail?** Yes, in three places.

1. Every shard sums `exp(x - M)` over all its columns, so the shard holding the maximum contributes
   `1 + t_s` to its local sum, and the all-reduced `global_sum = 1 + T` in FP32. `log(1 + T)` is then
   exactly 0 once `T < 2^-24` (one competitor at a gap of about 17 or more).
2. `log_normalizer = M + log(global_sum)` is formed before `x_t = M` is subtracted, so `log1p(T)` is lost
   below half an ulp of `|M|` even when (1) kept it: at `|M| = 32` that is `T < 2^-19` (a gap of about 13).
3. Backward: `scaled_softmax` is rounded to BF16 and `subtract_at_target` subtracts `inv_N` after that
   rounding: `dx_t = bf16(p_t s) - s`. With `p_t = 1 / (1 + T)`, `bf16(p_t s) = s` once `T < 2^-9`, so the
   target gradient is exactly 0 from a single-competitor gap of about 6.3.

**Extra hazards from the cross-device reduction.**

- *Precision of the all-reduced partial sums.* The partial sums are FP32 and reduced by the CCL
  (`all_reduce_async`: reduce-scatter + all-gather). Whatever its internal accumulation, the operand that
  matters already has the maximum's `1` folded in, so the reduction cannot preserve a tail below `2^-24`;
  fixing the reduction alone would not help. The fix sends no partial sum through a reduction at all.
- *Ties for the maximum across shards.* With an all-gathered maximum and no index, ties are harmless in
  the old code (both copies contribute 1, `sum >= 2`, no collapse). A fix that excludes "the maximum" per
  shard must exclude exactly one element globally: if every shard holding a copy of `M` excluded its own,
  `T` would lose those copies (loss `log(1) = 0` instead of `log 2` for two tied copies). The fix picks one
  global first maximum in vocabulary order (lowest shard, then lowest column) and counts every other copy in
  the tail.
- *Which shard owns the selected element.* The shard holding `k` is generally not the shard holding the
  target `t`. The loss needs only `x_t - M` and `log1p(T)`, so it can be formed on every device once the
  target logit is shared. The backward's Eq. 7 reconstruction (`dx_k = -sum_{j != k} dx_j`) would need a
  cross-shard sum, but for this loss it has a closed form, `s * expm1((x_t - M) - l)`, that only the
  target's shard evaluates; no communication is needed.

## Proof on hardware (before the fix)

`tests/ops/distributed/vocab_parallel_ce_collapse_test.cpp` (`VocabParallelCollapseTest`, committed before
the fix) opens a `[1, tp]` mesh (the largest of 8, 4, 2 the system holds), shards the logits on the vocabulary
along mesh axis 1 and calls the op with `cluster_axis = 1`, as nano_gpt does. Every value is compared with a
float64 oracle on the BF16 logits read back from the devices, with the log_softmax tests' BF16 relative
tolerance (`1e-2`) and no absolute slack beyond flushed denormals (`1e-30`); confident rows also require
`loss > 0` and `dx[t] < 0`, and every TP device's copy of the loss is checked.

- `ConfidentTargetKeepsLossAndGradient`: the target is the global maximum on each shard in turn, at the
  shard's first column, last column (shard boundaries), middle or a pseudo-random column; competitors trail
  by gaps 8, 9, 12 (BF16 cliff of `1 + T`) and 17, 20, 24 (FP32 cliff), on the same shard, on the next shard
  (across the boundary for an edge target) or all of them; offsets 0, -32, 32. Local widths `32000 / tp`
  (NanoLlama3), 96 (tile aligned) and 100 (padded tile per shard). NONE reduction with a 0.25 upstream.
- `MeanLossKeepsConfidentSignal`: MEAN reduction (the nano_gpt path) on rows whose tail is all competitors at
  gaps 17-24.
- `MaximaTiedAcrossShards`: two or three copies of the maximum on different shards (same column, including
  shard edges); the target is the first copy, a later copy or another column.
- `MixedRowsMatchFloat64`: random logits at `V = 32000`, a quarter confident rows, a quarter tied maxima with
  the target on the later copy, per-row upstream weights including zeros.

The logits are BF16 only (the op rejects FP32), so the FP32 gaps 17-24 test the FP32 accumulation and the
FP32 `M + log(sum)`, and the BF16 outputs can still represent losses down to `e^-24`.

Wormhole n300 (`[1, 2]` mesh, IRD wh-lb-45), old implementation: **3 of 4 tests fail**; the 10 existing
`ShardedCrossEntropyLossTest` tests pass (their `5e-2` absolute tolerances cannot see a 3e-4 loss become 0).

| Case | Mismatches (old) |
| --- | ---: |
| `[1,1,64,2x16000]` forward (each device) / backward | 109 / 110 |
| `[2,1,35,2x96]` forward (each device) / backward | 106 / 132 |
| `[1,1,64,2x100]` forward (each device) / backward | 118 / 119 |
| MEAN `[2,1,32,2x1024]`: mean loss / backward | 0 instead of 1.68e-5 / 128 |
| Mixed `[2,1,64,2x16000]` forward (each device) / backward | 64 / 189 |
| Ties across shards | 0 (passes) |

Every confident loss is exactly 0 from the smallest gap, 8 (`3.354e-4` expected, `0` actual), and every
confident target gradient is exactly 0 (for example `-8.38e-5` expected, `0` actual). The forward collapses
earlier than the FP32 cliffs above: the composite's FP32 `exp` / `sum` / `log` chain does not resolve
`1 + 3.4e-4` from 1. Most likely the FPU row reduction in `ttnn::sum` reads its FP32 operand through the
TF32 source registers (10 explicit mantissa bits, as found for the old fused kernel); this was not isolated
further, since the fix removes the sum of `1 + T` altogether. In the mixed test the old backward is also
more than 1% off on ordinary (non-target) gradients of random rows (for example `3.399e-8` for `3.363e-8`).

Blackhole LoudBox (8x p150, `[1, 8]` mesh with `tt-train/configs/mgd/bh_1_8_line_line.textproto`, Exabox
bh-lb-120-a07u28), the NanoLlama3 TP8 split (`32000 / 8 = 4000` columns per device): the same **3 of 4 tests
fail**, on all eight devices.

| Case (TP8) | Mismatches (old) |
| --- | ---: |
| `[1,1,64,8x4000]` forward (each device) / backward | 89 / 110 |
| `[2,1,35,8x96]` forward (each device) / backward | 102 / 120 |
| `[1,1,64,8x100]` forward (each device) / backward | 94 / 113 |
| MEAN `[2,1,32,8x1024]`: backward | 128 |
| Mixed `[2,1,64,8x4000]` forward (each device) / backward | 52 / 162 |
| Ties across shards | 0 (passes) |

On Blackhole the old forward still resolves the gap-8 row (`T = 3.4e-4`) and the MEAN test (whose mean is
dominated by its largest tails), but every confident loss from gap 9 is exactly 0; the target gradient is
exactly 0 from gap 8 on both architectures.

## Fix

Three new kinds of the fused row-softmax program (`metal/ops/log_softmax/row_softmax_program.{hpp,cpp}`,
same kernels as the fused log_softmax and cross-entropy, under `VP_*` defines), wrapped by
`metal/ops/log_softmax/vocab_parallel_cross_entropy.{hpp,cpp}` as one mesh-workload device operation (one
program per TP rank, which knows its vocabulary offset `first_v = rank * V / tp`):

1. **Shard statistics** (`VocabParallelStats`, per device): the fused log_softmax's online pass over the local
   columns gives per row the local maximum `m_s`, its first column `k_s` (made global by adding `first_v`) and
   the local tail `t_s` *without* that element, in FP32 (no log1p). The reader also fetches the target logit
   with one aligned 64-byte read when the target falls in this shard (otherwise `+0`). Output: one FP32 tile
   row per row with columns `(m_s, first_v + k_s, t_s, x_t or 0)`.
2. **All-gather** the records along the vocabulary axis: `[B, 1, S, 32 * tp]` FP32. An all-gather moves
   values; nothing is added in the collective.
3. **Combine** (`VocabParallelCombine`, identical on every device): the shards become the lanes of the same
   combine the fused kernels use for SFPU lanes and split rows: `M = max m_s`,
   `K = min{first_v + k_s : m_s = M}` (lowest shard on ties),
   `T = sum_s t_s exp(m_s - M) + sum_{s : lane != K} exp(m_s - M)`, never adding the selected `e^0`;
   `l = log1p(T)` in FP32; `loss = -((x_t - M) - l)`, exactly `l` when `t = K`. It also writes the row state
   `(M, l)`, two FP32 column-vector tiles per tile row, for the backward.
4. **Backward** (`VocabParallelBackward`, per device, local): `dx_j = exp(x_j - M) * q` with
   `q = s exp(-l)` from the saved state, one pass over the logits (no statistics recomputation, unlike the
   single-device backward), and the target's shard substitutes `dx_t = s * expm1((x_t - M) - l)`, which is
   `-s T / (1 + T)` (Eq. 7 in closed form) for `t = K`. The upstream gradient (`[1,1,1,1]` for MEAN,
   `[B,1,S,1]` for NONE) is folded into `s` in the kernel, as in the fused single-device backward.

Unchanged: the public API (`losses.hpp`), BF16-only logits, NONE/MEAN reductions (the MEAN sums stay
`ttnn::sum` in FP32), the out-of-range-target convention (zero target logit, no `-s` term), padding columns
(`-inf` in the reader, zero gradient), and `select_target_logit` / `subtract_at_target` (still public ops with
their own tests).

## Results

*Old* = the composite loss before this change (with this change's tests and benchmark), *fix* = this change;
same build per machine, only tt-train rebuilt between versions.

### Correctness

Wormhole n300, `[1, 2]` mesh (IRD wh-lb-45, `TT_VISIBLE_DEVICES=0`), with the fix: 4/4
`VocabParallelCollapseTest` and 10/10 existing `ShardedCrossEntropyLossTest`; and, for the shared kernels and
the remaining vocab-window ops, 73/73 of `CrossEntropyCollapseTest`, `CrossEntropyForwardTest`,
`CrossEntropyBackwardTest`, `FusedLogSoftmax*`, `SelectTargetLogitTest`, `SubtractAtTargetTest` and
`Mesh1x2WorkloadCacheHitTest`. No tolerance changed.

Blackhole LoudBox, `[1, 8]` mesh: 4/4 `VocabParallelCollapseTest` with the fix. The existing
`ShardedCrossEntropyLossTest` cannot run there with the 1x8 mesh descriptor forced (its fixture opens a
1x2 mesh and the fabric router times out), with the old and the new code alike; it ran on the Wormhole n300.

### Operation latency and DRAM

`ops::distributed::vocab_parallel_cross_entropy_loss(MEAN)` forward + backward, BF16 logits sharded on the
vocabulary, `cluster_axis = 1`, median ms (`tests/benchmark/vocab_parallel_ce_benchmark.cpp`: C++
`steady_clock` around eager dispatch with mesh synchronization). W is the full vocabulary; each device holds
W / tp. DRAM is per device: forward peak and step peak are the incremental DRAM of one forward and one forward
+ backward (the logits gradient itself is 2 bytes x B x S x W / tp: 524 MB at NanoLlama3 TP2).

TP2 (Wormhole n300):

| Shape `[B,1,S,W]` | old fw | fix fw | old bw | fix bw | old fw+bw | fix fw+bw | old step DRAM | fix step DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **[64,1,256,32000] NanoLlama3** | 20.93 | 13.38 | 31.45 | 10.37 | 52.35 | **23.54** | 1578 MB | **528 MB** |
| [8,1,256,32000] | 3.07 | 2.74 | 4.55 | 2.14 | 7.61 | 4.92 | 197 MB | 66 MB |
| [1,1,256,32000] | 0.90 | 0.52 | 1.20 | 0.24 | 2.09 | 0.76 | 24.7 MB | 8.3 MB |
| [1,1,32,32000] | 0.86 | 0.39 | 0.86 | 0.12 | 1.33 | 0.52 | 3.1 MB | 1.0 MB |
| [1,1,32,128256] | 1.27 | 0.50 | 3.27 | 0.21 | 4.53 | 0.72 | 12.3 MB | 4.1 MB |

NanoLlama3 forward peak: 1054 MB (old: the FP32 `exp(x - M)` of the whole shard) -> 11.5 MB; retained for
the backward: 3.1 MB -> 4.2 MB (the FP32 row state `(M, l)`, two tiles per tile row).

TP8 (Blackhole LoudBox, 8x p150, `[1, 8]` line):

| Shape `[B,1,S,W]` | old fw | fix fw | old bw | fix bw | old fw+bw | fix fw+bw | old step DRAM | fix step DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **[64,1,256,32000] NanoLlama3** | 4.02 | 1.99 | 3.71 | 0.96 | 7.56 | **2.88** | 398 MB | **135 MB** |
| [8,1,256,32000] | 1.28 | 0.79 | 0.53 | 0.22 | 1.19 | 0.86 | 49.8 MB | 16.9 MB |
| [1,1,256,32000] | 0.75 | 0.50 | 0.23 | 0.06 | 0.91 | 0.35 | 6.2 MB | 2.2 MB |
| [1,1,32,32000] | 1.01 | 0.48 | 0.36 | 0.06 | 1.17 | 0.52 | 0.78 MB | 0.28 MB |
| [1,1,32,128256] | 0.99 | 0.50 | 0.62 | 0.07 | 1.20 | 0.54 | 3.1 MB | 1.0 MB |

NanoLlama3 TP8 forward peak: 267 MB -> 23 MB. On Blackhole the small-shape forward times are bimodal from
sample to sample (for example 0.30-0.71 ms for `[1,1,32,32000]` with the fix, 0.98-2.17 ms before), which
is the collective's latency: a forward-only median can exceed the forward + backward one. With few rows the
loss is bound by the all-gather, not by the kernels.

Observations:

- The old forward materialized `exp(x - M)` for the whole shard in FP32 and the old backward made a BF16
  softmax, a separate target scatter and a broadcast multiply: three full-shard passes and three full-shard
  tensors. The fix reads the logits once forward and once backward and writes only the gradient.
- The backward no longer recomputes the statistics (the saved row state replaces them), which the
  single-device fused CE still does; it is 3x faster at NanoLlama3 TP2.
- The collective changed from one all-gather (BF16 maxima) and two all-reduces (FP32 sums, BF16 target logits)
  to one all-gather of a 4 KB FP32 record tile per tile row and shard.

### Tensor-parallel NanoLlama3 training

`nano_gpt` with `training_shakespeare_llama3_tensor_parallel.yaml` (`llama3.yaml`: batch 64, 6 blocks, d=384,
T=256, V=32000, AdamW 3e-4, weight tying) on Shakespeare tokenized with the TinyLlama tokenizer, 500 steps,
`--track_memory`. Step time is the median over steps 11-500.

TP2, Wormhole n300 (`mesh_shape: [1, 2]`, `TT_MESH_GRAPH_DESC_PATH=.../n300_mesh_graph_descriptor.textproto`):

| 500 steps | old | fix | change |
| --- | ---: | ---: | ---: |
| Median step time (seeds 5489 / 1 / 2) | 287.6 / 287.6 / 287.7 ms | **258.6 / 258.5 / 258.6 ms** | -10.1% |
| First-iteration cumulative DRAM peak | 3412 MB | **2480 MB** | -27% |
| Forward-pass segment peak | 2779 MB | 1785 MB | -994 MB |
| Mean loss, steps 401-500 (seeds 5489 / 1 / 2) | 3.2433 / 3.2248 / 3.2666 | 3.2448 / 3.2256 / 3.2686 | +0.0008 to +0.0020 |
| Final loss, step 500 (BF16) | 3.109 / 3.109 / 3.141 | 3.109 / 3.125 / 3.141 | |

The curves agree to the logged BF16 precision (a step of 0.0156 at a loss of 3.2) for the whole run; the
fixed run's 100-step mean is higher by 0.03-0.06% on all three seeds, a fraction of one logged BF16 step,
which these runs cannot separate from a change of rounding. At a loss of 3, almost no token is in the
collapse regime; the fix matters as the model becomes confident.

TP8, Blackhole LoudBox (`mesh_shape: [1, 8]`). The stock `training_shakespeare_llama3_tensor_parallel.yaml`
does not run on 8 devices (`llama3.yaml` has 6 heads, not divisible by the TP size), so these runs use
`llama3.yaml` with `num_heads: 8, num_groups: 8, embedding_dim: 512` and everything else unchanged; the loss
sees the same `[64, 1, 256, 32000]` logits, 4000 columns per device.

| 500 steps | old | fix | change |
| --- | ---: | ---: | ---: |
| Median step time (seeds 5489 / 1 / 2) | 142.2 / 142.1 / 142.4 ms | **137.2 / 137.4 / 137.2 ms** | -3.5% |
| First-iteration cumulative DRAM peak | 1716 MB | **1546 MB** | -10% |
| Forward-pass segment peak | 1470 MB | 1237 MB | -233 MB |
| Mean loss, steps 401-500 (seeds 5489 / 1 / 2) | 2.5719 / 2.5492 / 2.5973 | 2.5773 / 2.5494 / 2.6011 | +0.0002 to +0.0054 |
| Final loss, step 500 (BF16) | 2.281 / 2.328 / 2.328 | 2.297 / 2.328 / 2.328 | |

Across the six seed pairs (TP2 and TP8) the fixed run's late 100-step mean is never lower and is higher by
0.01-0.2%, at most a third of one logged BF16 step (0.0156 at 2.5). Two caveats on reading that as a
training difference: the logged value is the loss op's own output, and the old forward under-reports it
(every token whose loss falls below the old collapse threshold reads exactly 0, more of them as training
progresses), so part of the gap may be the measurement itself; and these runs cannot separate a
difference this small from a change of rounding. Not verified further. Training is not slower to converge in
any visible way: the curves agree to the logged precision over the whole run.

### Tensor-parallel TinyLlama training

`nano_gpt` with the stock `training_shakespeare_tinyllama.yaml` recipe (batch 5, AdamW 3e-4, warmup then
linear decay, 1000 steps) and `tinyllama_loudbox_tp.yaml` (1.1B: 22 blocks, d=2048, 32 heads, 8 KV groups,
T=2048, V=32000), `enable_tp: true`, on the Exabox Blackhole LoudBox: TP8 on all 8 p150s, TP2 on two of them
(`TT_VISIBLE_DEVICES=0,1`, `bh_1_2_line_line.textproto`). Batch 5 fits at both ranks.

| 1000 steps | TP8 old | TP8 fix | TP2 old | TP2 fix |
| --- | ---: | ---: | ---: | ---: |
| Loss op fw+bw at `[5, 1, 2048, 32000]` | 4.42 ms | **1.73 ms** | 14.63 ms | **4.56 ms** |
| Loss op step DRAM per device | 249 MB | **85 MB** | 986 MB | **330 MB** |
| Median step time (steps 11-1000) | 812.4 ms | **809.6 ms** (-0.3%) | 1294.7 ms | **1284.6 ms** (-0.8%) |
| First-iteration cumulative DRAM peak | 2784 MB | 2784 MB | 5918 MB | 5918 MB |
| Mean loss, steps 901-1000 | 0.0731 | 0.0740 | 0.0745 | 0.0745 |

TP8 is the mean of seeds 5489, 1 and 2; TP2 is seed 5489. The step saves what the op saves (2.9 vs 2.7 ms at
TP8, 10.1 vs 10.1 ms at TP2). The step's DRAM peak does not move, so it is reached outside the loss.

1000 steps of 5 x 2048 tokens is about 28 passes over the 368k-token set, so the model memorises it and the
loss falls to about 0.07: most tokens are confident. The runs are bit-reproducible (a repeat of old TP8 seed
5489 matched all 1000 logged losses), so the old and fixed runs diverge only because the fix's arithmetic
differs. They first log a different loss at steps 5-40. After that they differ by 3-4% per step on average,
in both directions. The late mean changes by +1.1%, +3.2% and -0.6% (TP8 seeds) and -0.05% (TP2). The old
runs' late means span 0.069-0.080 across seeds.

## Hardware notes

- IRD Wormhole LoudBoxes (`wormhole_b0 --model lb`, wh-lb-45 and wh-lb-47): only the two chips of each n300
  are linked. Fabric auto-discovery downgrades to a 2x1 mesh and an 8-chip mesh fails (the T3K descriptor has
  "no embeddable MGD fallback"), so they give TP2 only, with `TT_VISIBLE_DEVICES=0`.
- After a crashed mesh run, reset the boards (`tt-smi -r 0,1,2,3`) before the next one.
- nano_gpt on one n300 needs `mesh_shape: [1, 2]` and
  `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto`
  (auto-discovery reports `[2, 1]` and nano_gpt rejects the mismatch).
- Exabox Blackhole LoudBox: `TT_MESH_GRAPH_DESC_PATH=tt-train/configs/mgd/bh_1_8_line_line.textproto` for
  the `[1, 8]` mesh (tt-train's default for 8 devices is the Wormhole T3K descriptor).

## Reproduction

```bash
./build_metal.sh --build-tt-train
export TT_METAL_HOME="$PWD" TT_METAL_RUNTIME_ROOT="$PWD"
build_Release/tt-train/tests/ttml_tests --gtest_filter='VocabParallelCollapseTest.*:ShardedCrossEntropyLossTest.*'
build_Release/tt-train/tests/vocab_parallel_ce_benchmark   # [1, tp] mesh, tp = 8, 4 or 2: the largest visible
build_Release/tt-train/sources/examples/nano_gpt/nano_gpt -c <tensor-parallel config with data_path> --track_memory
```

To measure the old implementation, build the tests and benchmark on the parent revision with the old
`losses.cpp`.
