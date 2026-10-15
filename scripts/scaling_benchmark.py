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

    python3 scripts/scaling_benchmark.py --devices 1,2,4,8 --copies 16
        The real thing. Needs Boltz installed and that many visible GPUs. Prints wall-clock,
        speedup against the 1-device run, and parallel efficiency.

    python3 scripts/scaling_benchmark.py --devices 1 --copies 2 --samples 1
        A cheap smoke test that the invocation works at all before spending a node on it.

Reading the output: efficiency is speedup divided by device count. Below about 60% at 8
devices, the fixed costs per device -- model load, input parsing -- are dominating, and the
answer is a larger batch rather than more cards. That is the number worth reporting back.
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

# Public. GFP, UniProt P42212, the sequence every structural biology course uses. It is here
# so this script needs no input file and leaks nothing: a reviewer can verify the sequence
# against UniProt in a browser.
GFP = ("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
       "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
       "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK")

SUBS = "ACDEFGHIKLMNPQRSTVWY"


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
    ap.add_argument("--copies", type=int, default=16,
                    help="independent folds in the batch (default 16). Keep it a multiple of "
                         "the largest device count, or the last device idles and efficiency "
                         "reads low for the wrong reason.")
    ap.add_argument("--samples", type=int, default=1,
                    help="diffusion samples per fold (default 1 -- this is a timing run)")
    ap.add_argument("--out", default="bench_out", help="working directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit. No GPU, no Boltz, no network.")
    a = ap.parse_args()

    counts = [int(x) for x in a.devices.split(",") if x.strip()]
    out = Path(a.out).resolve()
    visible = gd.visible_gpu_count()

    print("=" * 78)
    print("  FOLD BATCH SCALING -- %d independent folds, %d diffusion sample(s)"
          % (a.copies, a.samples))
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

    (out / "scaling.json").write_text(json.dumps(
        {"copies": a.copies, "samples": a.samples, "visible_gpus": visible, "rows": rows},
        indent=2))
    print("  wrote %s" % (out / "scaling.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
