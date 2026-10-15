# Batch scripts

Two job scripts, for a Slurm cluster.

| file | what it does | runnable from this repository? |
|---|---|---|
| `scaling.sbatch` | times the same panel at 1, 2, 4 and 8 devices and writes the speedup table | **yes**, with Boltz installed |
| `campaign.sbatch` | the production shape: one design campaign across 8 devices | no — needs the pipeline, which is not public |

## Both are untested

Neither has been submitted to a real scheduler, because there is no cluster here. They are
included because "how would you use the allocation" deserves a job script rather than a
paragraph, and because the shape of the job is part of what is being proposed. The lines that
will certainly need editing for a given site are marked `EDIT`: account, partition, the module
loads, and the `--gres` syntax, which differs between sites more than it should.

Treat them as a design, reviewable as a design. Everything in this repository that *is* tested
is tested by `selftest.py`, and these two files are not part of it.

## The one thing in here that is not boilerplate

A batch script normally repeats the device count: once for the scheduler, once for the
application. These do not. `--gres=gpu:8` asks Slurm for hardware; the `@GPU devices=8` line in
the loadfile declares what the run intends to use. **If the two disagree the run is refused and
both numbers are named** — it does not pick one and proceed.

That is worth more than it looks, because of what Slurm does on a shared node. Granted two
cards out of eight, Slurm tells you so by setting `CUDA_VISIBLE_DEVICES=3,5` while `nvidia-smi`
goes on truthfully reporting eight. Code that asks the hardware how many GPUs exist gets the
wrong answer in the only environment that matters: it would let a `devices=8` request through,
hand the fold engine eight, and reach for six cards belonging to someone else's job.

`allocated_gpu_count()` in `scripts/gpu_devices.py` reads the scheduler's variable first and
falls back to the hardware only when there is no allocation to respect. So:

```
GPUs granted : 3,5        <- the allocation. This is what governs.
GPUs on node : 8          <- the inventory. Printed, and deliberately ignored.
```

`scaling.sbatch` prints both lines next to each other for exactly this reason, and they are
allowed to differ.

This was a genuine bug, found while writing these scripts, and it is the kind that cannot
surface on the hardware the rest of this repository was developed on: a one-card workstation
has no subset to be given. The tests for it are in `gpu_devices.py --selftest`, simulating a
partial allocation on an eight-card node.

## Before spending a node

```bash
python3 scripts/gpu_devices.py --plan ../example_8gpu.loadfile.txt
```

Resolves the pairing anywhere, including on a login node with no GPU, and costs a second. A
mismatch caught here is a typo; the same mismatch caught after `sbatch` is a wasted node-hour.
