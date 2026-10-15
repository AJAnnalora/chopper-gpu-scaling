# Multi-GPU device policy — test code

This repository holds the parts of a protein-design pipeline that decide how the hardware gets
used: how many GPUs a run may use, how the work is split across them, what command line that
produces, and how jobs take turns on a machine so they never collide. Two modules and their
tests. The pipeline itself is not here and is not public.

It exists because a compute-allocation request should be checkable. Below is how to check it
in under a minute, on any machine, **with no GPU and nothing to install.**

**You may read it and run it; you may not reuse it.** That is deliberate and it is what
`LICENSE` says — the copyright position on this code is not settled yet, so no rights are
granted. Reading and running is all that reviewing requires, and it is explicitly permitted.
If you want to use any of it, ask.

## The honest status, first

**Nothing in this repository has ever executed on more than one device.** This laboratory has
a single RTX 3090, and the pipeline's own safety rule holds it to one job at a time. Device
count and split mode are declarable parameters; multi-device execution is **plumbed and
awaiting hardware.**

That is the thing the allocation would change, so it would be strange to claim otherwise.
What is offered here instead is the code that will do the work, the policy that governs it,
and tests that simulate the hardware we do not have.

## Check it

Requires Python 3 and nothing else — no `pip install`, no GPU, no network.

```bash
python3 selftest.py
```

Both modules' suites, around fifty assertions. They simulate a 1-GPU box and an 8-GPU box,
and they simulate crashed and competing jobs, checking the behavior that actually matters when
an allocation is on the line.

Then see a multi-GPU run resolved on whatever machine you are sitting at:

```bash
python3 scripts/gpu_devices.py --plan example_8gpu.loadfile.txt
```

On a single-GPU machine that prints:

```
@GPU  requested=8  visible=1  using=1  split=batch   plan only; requested 8, this box has 1
  on this box (1 visible) : boltz predict ... --devices 1 --num_workers 4
  as requested (8 GPUs)   : boltz predict ... --devices 8 --num_workers 4
  ^ the second line is what runs where 8 devices exist. Identical loadfile, nothing toggled.
```

And what this machine reports for itself:

```bash
python3 scripts/gpu_devices.py --show
```

## The four decisions worth checking

**It refuses rather than silently downgrades.** Requesting 8 devices and *executing* with one
visible is refused outright, naming both numbers. It does not quietly run at 1 and report
success. An allocation burned by code that ran at one-eighth speed and said nothing is the
failure this prevents.

**The gate is split by intent, not by a demo flag.** *Planning* a run never fails on device
count — an 8-device request resolves and prints honestly on a 1-device box, which is how the
output above is possible. *Executing* enforces the count exactly. There is no
`--pretend-i-have-8` switch, because a demonstration that works only because a flag is set is
a demonstration that breaks on the demonstration machine. The same file, unchanged, requests 8
in both places.

**An undetermined device count is refused, not guessed.** If the GPU count cannot be
established, a multi-device request fails rather than assuming a number.

**Unsupported paths are marked unsupported.** The capability map records, per engine, whether
a split mode works:

| engine | `batch` | `context` |
|---|---|---|
| `boltz` | SUPPORTED | UNSUPPORTED |
| `af2` | UNSUPPORTED | UNSUPPORTED |

AlphaFold-2 is `UNSUPPORTED` on purpose. Its multi-GPU story is not device sharding, and a
flag that merely parses is not support. If AF-2 sharding is wanted later it earns its own
status rather than inheriting one.

## Measure it yourself

The honest limit of everything above is that it was written on one GPU. So rather than claim a
scaling factor, this repository ships the thing that measures one:

```bash
python3 scripts/scaling_benchmark.py --dry-run          # what it would run. No GPU, no install
python3 scripts/scaling_benchmark.py --devices 1,2,4,8 --copies 16
```

The real run needs [Boltz](https://github.com/jwohlwend/boltz) — the open folding engine this
pipeline drives — and that many visible GPUs. It folds 16 independent copies of a public
sequence (GFP, UniProt P42212, one substitution apart so none is a cache hit), shards them
across the devices, and prints wall-clock, speedup against the single-device run, and parallel
efficiency. Results are written to `scaling.json`.

It is **offline**: the inputs declare an empty alignment, so nothing is sent to an MSA server.
That keeps network variance out of a timing measurement and keeps the sequences on the machine.

**The device arguments are built by `scripts/gpu_devices.py`** — the module the tests cover —
rather than assembled separately here. If the policy and the benchmark could disagree, the
benchmark would be measuring something other than what a run does.

### What this does not measure

Stated as its own heading because it is the sentence that keeps the rest honest, and a caveat
buried mid-paragraph is a caveat nobody reads.

**It measures sharding, not campaign throughput.** Two reasons. It folds *single chains*,
where a real campaign folds two-chain complexes, which are larger and slower per fold. And it
measures the task-parallel case only — independent folds spread across devices. It says nothing
about context-parallel folding of one oversized complex, which no installed engine supports
(see above).

So a good number here means the sharding works. It does not mean a campaign runs this fast.

If efficiency falls below roughly 60% at 8 devices, per-device fixed costs — model load, input
parsing — are dominating, and the answer is a larger batch rather than more cards. That is the
number we would most like to know and cannot currently produce.

---

## Taking turns on the hardware

Deciding how many devices a run may use is half the problem. The other half is making sure two
runs never try to use the same ones, and that a crashed run does not leave the machine idle.
`scripts/chopper_launch.py` is that half:

```bash
python3 scripts/chopper_launch.py --status      # what is running, waiting, finished
python3 scripts/chopper_launch.py --selftest    # the tests described below
```

**One job at a time, and it is a lock rather than a promise.** The lockfile is created with
`O_EXCL`, so two launches racing cannot both win — one takes it, the other is refused and told
what is already running. A policy written in a comment gets forgotten; a policy enforced by the
filesystem does not.

**A crashed job does not wedge the queue.** The lock holds the process id of whatever took it.
If that process is gone, the lock is detected as stale by asking the operating system — and
distinguishing a dead process from a zombie, which `kill(pid, 0)` alone does not do — then
cleared, and the interrupted job is recorded as interrupted rather than quietly forgotten.

**The queue keeps the hardware busy unattended.** Jobs can be queued with priorities, the same
job cannot be queued twice, a job already running will not be queued again, and asking the
queue to start something while a job is running is refused rather than allowed to overlap.

Those are the properties the test suite asserts, including the cases that are awkward to
produce by hand: a dead job's lock, a lock inherited after a crash, a pump attempted while
busy, and a queue surviving all of it.

### Why this belongs in a compute-allocation discussion

The failure it prevents is not exotic. A shared machine, two runs started a minute apart, both
grabbing the same devices, both finishing slower than either would alone, and no record
afterwards of which pose came from which. Or the quieter version: a job dies at 2am and the
hardware sits idle until someone notices. Neither is a modeling problem, and both waste the
thing being allocated.

Two of the test names in here are the scars of that: a job already running is not queued again,
and an interrupted job is recorded rather than forgotten.

---

## Two modes, and only one of them works today

A design campaign needs parallelism of two different kinds, and they are not interchangeable.

**`split=batch` — task-parallel, live and tested here.** The scaling axis is the number of
independent candidates, not the size of one system. A generative run proposes *n* sequences
from one seed; each is folded against the same target with the same cached alignment, and no
candidate depends on another's result. Nothing is exchanged between devices, so this
distributes across nodes without modification and is restart-safe. Sharding the candidate list
should scale close to linearly, with per-device model load and cache reads as the fixed cost —
so efficiency falls off at small *n* and improves as panels get larger. Most of a campaign is
this shape.

**`split=context` — model-parallel, and NOT supported.** A single complex can exceed one
card's memory: a binder against a receptor dimer with its natural ligand present runs to
600–900 residues, and that is where a single GPU's memory ceiling binds. Sharding one model
across tightly coupled devices is the answer, and `batch` does not provide it — adding more
cards does not make one model fit on one card.

### The mode that is not supported yet

The capability map marks `context` as `UNSUPPORTED` for both engines, and that is a real gap,
not a formality. Neither engine this pipeline currently drives can shard a single model across
devices. The declaration parses, and an attempt to execute it is refused rather than quietly
run as something else.

The route out is named in the source rather than left vague — `gpu_devices.py` carries it as a
comment beside the map:

```python
# when Fold-CP (arXiv:2603.14806) is installed in its OWN env and gated:
# "boltz_cp": {"batch": "SUPPORTED", "context": "GATED"}
```

A context-parallel folding engine, installed in its own environment and entering the registry
`GATED` rather than trusted. That is work to be done, not a capability to be claimed, and the
table says `UNSUPPORTED` until it is done and measured.

Stating this plainly is the point of publishing the map at all. A pipeline whose capability
table lists only successes is a table that tells a reader nothing.

### A measured baseline

On the single RTX 3090, one two-chain complex at 5 poses — fold, score and report — takes
**1 min 52 s** wall clock at 5.6 GB peak memory. That is about 22 s per pose end to end.
A larger campaign on the same hardware measured **523 poses/hour**, or ~7 s per pose: the
difference is the per-run fixed cost, which is paid once and amortizes over a panel. Both
numbers are from this hardware; neither is a projection.

The first thing worth measuring on real multi-GPU hardware is therefore scaling efficiency at
2, 4 and 8 devices on one fixed panel — which would also be the first chance to find out
whether the near-linear expectation above survives contact with a real node.

## Files

```
scripts/gpu_devices.py      the whole device-policy surface: @GPU parsing, the capability
                            map, resolution, the plan/execute gate, and the engine argv
scripts/chopper_launch.py   the job lock and queue: one job at a time, enforced
scripts/scaling_benchmark.py  measure real speedup across N GPUs. --dry-run needs neither
example_8gpu.loadfile.txt   a declaration asking for 8 devices. Device policy only --
                            no target, no sequences
selftest.py                 runs every test in the repository
LICENSE                     no license granted; shared for review
```

Both modules import only the Python standard library — no array or tensor libraries, no
pipeline modules. That is deliberate: the gate has to fail in a second rather than forty
minutes into a fold with a model stack loaded, and it means anyone can run these tests.

## Scope

This is an extract, published for review of one specific claim. The design pipeline it belongs
to is not public and no part of it is included here — no targets, no sequences, no scoring, no
results. Nothing here has been validated at a laboratory bench, and nothing here is a
biological claim.
