# Measured results

Everything in here was produced by `scripts/scaling_benchmark.py` on the hardware this project
actually has: **one RTX 3090.** Nothing in this directory is a projection unless the file says
so, and the one thing that is a projection says so in its own field name.

| file | what |
|---|---|
| `scaling_rtx3090_1gpu.json` | the benchmark's own output, one device, 4 folds — proof the harness runs |
| `batch_size_curve_rtx3090.json` | wall-clock at 1, 2, 4 and 8 folds, and the cost model fitted to it |

## The number that matters

A batch of *K* independent folds on one card:

| folds | wall clock | per fold |
|---|---|---|
| 1 | 34.2 s | 34.2 s |
| 2 | 41.9 s | 21.0 s |
| 4 | 64.3 s | 16.1 s |
| 8 | 87.4 s | 10.9 s |

A straight line through those four points, R² = 0.98:

```
seconds  =  28.2  +  7.7 x folds
            ^^^^     ^^^
            paid once      paid per fold
            per run
```

**The fixed cost is about four times the marginal cost of one fold.** Model load, weight
transfer to the card, input parsing — paid once however many folds follow. That single fact
decides the entire multi-GPU question for this pipeline, because batch sharding gives each
device a share of the folds and **every device pays the 28.2 s separately.**

## What that predicts for 8 GPUs

Each device folds *K/N* of the panel and pays the fixed cost once, so
`t(K,N) = 28.2 + 7.7·K/N`:

| panel size | predicted efficiency at 8 devices |
|---|---|
| 16 | 43% |
| 32 | 58% |
| 64 | 72% |
| 128 | 84% |
| 256 | 91% |
| 868 | 97% |

**Predictions, from single-card measurements.** They ignore contention for host memory
bandwidth, PCIe and the filesystem, every one of which pushes the real number down — so read
the column as a ceiling. The point of publishing it is that it is falsifiable: run
`scripts/scaling_benchmark.py` on 8 real cards and the table is either approximately right or
it is wrong, and either way everyone learns something.

Two things follow, and they are the honest answer to "do you need the GPUs":

**Small panels do not justify multiple cards, and we say so.** At 16 or 32 candidates most of
the gain is eaten by eight copies of a 28-second startup. Anyone benchmarking this with a small
batch will measure poor efficiency, and that number will be *correct* — about the batch, not
about the sharding. The benchmark now defaults to 16 folds per device for exactly this reason
and prints a warning if asked for less.

**Our campaigns are the size where it does pay.** The panels this pipeline actually runs are
hundreds of candidates — the largest to date was 868 — which is where the fixed cost amortizes
and predicted efficiency reaches the 90s. That is the regime the request is about.

## A second, independent anchor

From a different, larger run on the same card: a 868-candidate campaign sustained **523
poses/hour**, about 7 s per pose, against **22 s per pose** for a single-complex run of the same
pipeline end to end. Same conclusion by a different route — per-run fixed costs dominate at
small *n* and amortize across a panel — measured on the full pipeline including scoring and
reporting rather than on bare folds.

## Scope

Single chains, one diffusion sample, empty MSA, no network. A real campaign folds two-chain
complexes, which are larger and slower per fold, so the absolute seconds here are not campaign
throughput. The *shape* — large fixed cost, modest marginal cost — is the transferable part, and
it is the part the allocation question turns on.
