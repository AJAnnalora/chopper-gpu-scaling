#!/usr/bin/env python3
"""scaling_benchmark.py -- measure how a fold batch actually scales across GPUs.

WHY THIS IS HERE. The rest of this repository shows how a run DECLARES the hardware it wants
and how jobs take turns on it. It cannot show multiple GPUs working, because the machine it
was written on has one card. This script closes that gap in the only honest way available:
it does not claim a scaling factor, it measures one, on whatever machine you run it on.

If you have 8 GPUs, you can find out in one command whether the near-linear expectation in
README.md survives contact with real hardware. We would like to know too.

WHAT IT RUNS. Boltz (https://github.com/jwohlwend/boltz), the open folding engine this
pipeline drives, invoked exactly as the pipeline invokes it -- the device arguments are built
by scripts/gpu_devices.py, the same module the tests cover, not reassembled here. The inputs
are K independent copies of a public sequence (GFP, UniProt P42212, with one substitution per
copy so they are distinct jobs rather than a cache hit). Nothing in this script is specific
to any target we work on, and no unpublished sequence appears in it.

WHAT IT MEASURES, AND WHAT IT DOES NOT. K independent folds sharded across N devices: the
task-parallel case, which is most of a real campaign. It does NOT measure context-parallel
folding of one large complex -- no installed engine supports that yet, which README.md says
plainly. It also folds single chains rather than two-chain complexes, so treat the numbers as
indicative of the sharding, not as our campaign's throughput.

USAGE

    python3 scripts/scaling_benchmark.py --dry-run
        Print exactly what would run, at each device count. No GPU, no Boltz, no network.
        Start here: it is also how you check what this will do before letting it near a node.

    python3 scripts/scaling_benchmark.py --devices 1,2,4,8
        The real thing. Needs Boltz installed and that many visible GPUs. Prints wall-clock,
        speedup against the 1-device run, and parallel efficiency. The batch size defaults to
        16 folds per device, which is large enough that the fixed startup cost each device pays
        does not dominate the result -- pass --copies to override, and read the note on
        PER_DEVICE_DEFAULT before choosing something small.

    python3 scripts/scaling_benchmark.py --devices 1 --copies 2 --samples 1
        A cheap smoke test that the invocation works at all before spending a node on it.

Reading the output: efficiency is speedup divided by device count. Low efficiency at a high
device count means the per-device fixed costs -- model load, input parsing -- are dominating,
and the answer is a larger batch rather than more cards.

HOW LOW IS TOO LOW depends on the batch, and results/README.md gives the measured prediction
instead of a rule of thumb: on this hardware a fold batch is 28.2 s of fixed cost plus 7.7 s
per fold, which puts 8 devices at roughly 84% efficiency for 16 folds per device and roughly
43% for 2. A result far BELOW the prediction for the batch actually used is the interesting
outcome, and the one worth reporting back.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gpu_devices as gd          # noqa: E402  -- the module under test builds the argv

BOLTZ_ENV = os.environ.get("BOLTZ_ENV", "boltz2")

#: Folds per device when --copies is not given. NOT an arbitrary round number: on the one card
#: available here a fold batch costs 28.2 s of fixed setup plus 7.7 s per fold (measured, four
#: batch sizes, R^2 = 0.98 -- results/batch_size_curve_rtx3090.json). Sharding only pays once
#: the per-device work is large against that 28.2 s, because every device pays it separately.
#:
#: At 16 folds TOTAL across 8 devices each device does 2 folds, pays 28.2 s to do 15 s of work,
#: and the measurement reports about 43% efficiency -- a true number about a badly chosen batch,
#: which a reader would reasonably mistake for a statement about the sharding. 16 folds PER
#: DEVICE predicts about 84%. So the default scales with the device count instead of being
#: fixed, and the hostile case is reachable on purpose with an explicit small --copies.
PER_DEVICE_DEFAULT = 16

# Public. GFP, UniProt P42212, the sequence every structural biology course uses. It is here
# so this script needs no input file and leaks nothing: a reviewer can verify the sequence
# against UniProt in a browser.
GFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
       "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
       "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")

SUBS = "ACDEFGHIKLMNPQRSTVWY"


#: Fitted on this box: four batch sizes (1, 2, 4, 8 folds) on one RTX 3090, R^2 = 0.98.
#: results/batch_size_curve_rtx3090.json holds the raw numbers.
FIXED_S, PER_FOLD_S = 28.2, 7.7


def predicted_efficiency(copies, devices):
    """What the single-card cost model expects, so a real result has something to be compared to.

    Batch sharding gives each device copies/devices folds, and EVERY device pays the fixed cost
    -- that is the whole of the model and the whole reason small batches scale badly:

        t(K, N) = FIXED + PER_FOLD * K/N          speedup = t(K,1) / t(K,N)

    It ignores contention for host memory bandwidth, PCIe and the filesystem, all of which push
    the real number DOWN. So treat it as a ceiling rather than a target: landing near it means
    sharding is working, and landing far below it is the finding worth chasing.
    """
    if devices <= 1 or copies <= 0:
        return 1.0
    t1 = FIXED_S + PER_FOLD_S * copies
    tn = FIXED_S + PER_FOLD_S * (copies / float(devices))
    return (t1 / tn) / devices


def variants(k):
    """k distinct sequences, one substitution apart. Distinct so nothing is cached twice."""
    out = []
    for i in range(k):
        pos = 20 + (i * 7) % 120            # inside the beta barrel, away from the termini
        aa = SUBS[i % len(SUBS)]
        seq = GFP[:pos] + (aa if GFP[pos] != aa else SUBS[(i + 1) % len(SUBS)]) + GFP[pos + 1:]
        out.append(("gfp_%02d" % i, seq))
    return out


def write_inputs(d, k):
    """One FASTA per job, in Boltz's chain-header form. Returns the input directory."""
    d.mkdir(parents=True, exist_ok=True)
    for name, seq in variants(k):
        (d / ("%s.fasta" % name)).write_text(">A|protein|empty\n%s\n" % seq)
    return d


def boltz_argv(inp, out, devices, samples):
    """The command, with its device arguments built by the module this repository tests.

    gpu_devices.resolve_request is what a real run calls; passing executing=False keeps it
    from refusing on a box with fewer cards, so --dry-run works anywhere.
    """
    res = gd.resolve_request("boltz", {"devices": devices, "split": "batch", "declared": True},
                             executing=False, visible=gd.visible_gpu_count())
    launcher = ([str(Path(os.path.expanduser("~/miniconda3/envs/%s/bin/boltz" % BOLTZ_ENV)))]
                if Path(os.path.expanduser("~/miniconda3/envs/%s/bin/boltz" % BOLTZ_ENV)).exists()
                else ["conda", "run", "--no-capture-output", "-n", BOLTZ_ENV, "boltz"])
    argv = launcher + ["predict", str(inp), "--output_format", "pdb", "--out_dir", str(out),
                       "--accelerator", "gpu"]
    # requested, not used: a benchmark asked for 8 must ASK for 8, or it measures nothing.
    argv += ["--devices", str(res["requested"])]
    # No --use_msa_server. The FASTA headers already declare an empty MSA, so asking a
    # public server for one would contradict them, add network variance to a timing
    # measurement, and send sequences off the machine. This benchmark is offline.
    argv += ["--diffusion_samples", str(samples)]
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default="1,2,4,8",
                    help="comma-separated device counts to time (default 1,2,4,8)")
    ap.add_argument("--copies", type=int, default=None,
                    help="independent folds in the batch. Default is %d x the largest device "
                         "count, so every device gets a full share and the fixed startup cost "
                         "is amortized rather than measured. A small batch makes any sharding "
                         "look bad -- see results/README.md, which has the measured numbers."
                         % PER_DEVICE_DEFAULT)
    ap.add_argument("--samples", type=int, default=1,
                    help="diffusion samples per fold (default 1 -- this is a timing run)")
    ap.add_argument("--out", default="bench_out", help="working directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit. No GPU, no Boltz, no network.")
    a = ap.parse_args()

    counts = [int(x) for x in a.devices.split(",") if x.strip()]
    if not counts:
        print("--devices was empty; nothing to time.")
        return 2
    out = Path(a.out).resolve()
    visible = gd.visible_gpu_count()

    if a.copies is None:
        a.copies = PER_DEVICE_DEFAULT * max(counts)
        derived = "  (%d per device x %d devices -- see PER_DEVICE_DEFAULT)" % (
            PER_DEVICE_DEFAULT, max(counts))
    else:
        derived = ""
        per_dev = a.copies / float(max(counts))
        if per_dev < 8:
            # Say it BEFORE the run rather than leaving a low number to be misread after it.
            print("NOTE: %d folds across %d devices is %.1f folds per device. Each device pays "
                  "the fixed\n      startup cost once, so a batch this small reports low "
                  "efficiency about the BATCH,\n      not about the sharding. Drop --copies to "
                  "let it scale with the device count."
                  % (a.copies, max(counts), per_dev))

    print("=" * 78)
    print("  FOLD BATCH SCALING -- %d independent folds, %d diffusion sample(s)%s"
          % (a.copies, a.samples, derived))
    print("  visible GPUs: %s" % ("undetermined" if visible is None else visible))
    print("=" * 78)

    if a.dry_run:
        for n in counts:
            print("\ndevices=%d" % n)
            print("  " + " ".join(boltz_argv(out / "in", out / ("d%d" % n), n, a.samples)))
        print("\nDRY RUN -- nothing was run and no input was written.")
        if visible is not None and counts and max(counts) > visible:
            print("Note: the largest count asked for (%d) exceeds the %d GPU(s) visible here. "
                  "A real run would refuse that; this plan does not."
                  % (max(counts), visible))
        return 0

    have_conda = shutil.which("conda") is not None
    have_direct = Path(os.path.expanduser("~/miniconda3/envs/%s/bin/boltz" % BOLTZ_ENV)).exists()
    if not (have_conda or have_direct):
        print("\nBoltz was not found. This benchmark needs it; the rest of this repository "
              "does not.\n  Install:  https://github.com/jwohlwend/boltz\n"
              "  Or set BOLTZ_ENV to the conda environment that has it.\n"
              "  Meanwhile, --dry-run shows exactly what would be run.")
        return 2
    if visible is not None and max(counts) > visible:
        print("\nRefusing: --devices asks for %d but %d GPU(s) are visible. Lower it, or run "
              "on a node that has them. (--dry-run plans any count anywhere.)"
              % (max(counts), visible))
        return 2

    inp = write_inputs(out / "in", a.copies)
    rows, base = [], None
    for n in counts:
        argv = boltz_argv(inp, out / ("d%d" % n), n, a.samples)
        print("\n--- devices=%d ---\n%s" % (n, " ".join(argv)))
        t0 = time.time()
        rc = subprocess.run(argv).returncode
        dt = time.time() - t0
        if rc != 0:
            print("  FAILED (exit %d) after %.1fs -- stopping. Partial results below." % (rc, dt))
            rows.append({"devices": n, "seconds": round(dt, 1), "ok": False})
            break
        if base is None:
            base = dt
        rows.append({"devices": n, "seconds": round(dt, 1), "ok": True,
                     "speedup": round(base / dt, 2),
                     "efficiency": round((base / dt) / n, 3)})
        print("  %.1fs" % dt)

    print("\n" + "=" * 78)
    print("  %-9s %-12s %-10s %s" % ("devices", "wall (s)", "speedup", "efficiency"))
    for r in rows:
        if r.get("ok"):
            print("  %-9d %-12.1f %-10.2f %.0f%%"
                  % (r["devices"], r["seconds"], r["speedup"], r["efficiency"] * 100))
        else:
            print("  %-9d %-12.1f %s" % (r["devices"], r["seconds"], "FAILED"))
    print("=" * 78)
    print("  %d folds, %d sample(s) each. Efficiency is speedup / devices." % (a.copies, a.samples))
    print("  Low efficiency at high device counts means per-device fixed costs dominate;")
    print("  the answer to that is a bigger batch, not more cards.")
    for n in counts:
        if n > 1:
            print("  predicted at %d devices, this batch: %.0f%%  (28.2s fixed + 7.7s/fold, "
                  "measured on 1 card)" % (n, 100.0 * predicted_efficiency(a.copies, n)))

    (out / "scaling.json").write_text(json.dumps(
        {"copies": a.copies, "samples": a.samples, "visible_gpus": visible,
         "folds_per_device": {str(n): round(a.copies / float(n), 2) for n in counts},
         "predicted_efficiency": {str(n): round(predicted_efficiency(a.copies, n), 3)
                                  for n in counts},
         "rows": rows},
        indent=2))
    print("  wrote %s" % (out / "scaling.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
