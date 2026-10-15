#!/usr/bin/env python3
"""gpu_devices.py -- DEVICE POLICY LIVES HERE.

@GPU parsing, the splits capability map, resolve_request() and validate_gpu_request() are the
ONLY device-policy surface. If you are about to add a GPU check somewhere else, you are
creating the second source of truth this comment exists to prevent. Edit here, nowhere else.

Implements docs/SPEC_gpu_split_directive.md  with ONE amendment, stated plainly
because it changes a documented rule:

  SPEC §3b rule 3 says `devices > 1` must FAIL when it exceeds the visible GPU count.
  That is right for a real run and wrong for planning one. The PI, 2026-10-05: "we need to be
  able to simulate it here, and they can actually run on 8 during the demo."

  So the gate is split by INTENT, not by a demo flag:
     executing=False (plan / --dry-run)  -> never fails on device count. Reports
                                            requested vs visible and emits the real
                                            multi-GPU invocation. This is the simulation.
     executing=True  (about to fold)     -> enforces the SPEC rule exactly, naming both
                                            numbers.
  A demo that works because a special flag is set is a demo that breaks on the demo box.
  There is no such flag here: the same loadfile, unchanged, requests 8 in both places and
  the resolver reports honestly what it actually got.

Stdlib ONLY -- no torch, no numpy. Two reasons: the gate must fail in a second rather than
forty minutes into a fold with the model stack loaded (SPEC §3b), and Chopper's no-GPU
verification paths claim zero third-party dependencies.

Nothing here has ever executed on more than one device: this box has one RTX 3090. The
honest status line stays the one from SPEC §8 -- "device count and split mode are declarable
funnel parameters; multi-device execution is plumbed and awaiting hardware" -- NOT
"we support multi-GPU".

CLI:  gpu_devices.py --selftest              # no GPU required
      gpu_devices.py --show                  # what this box reports
      gpu_devices.py --plan <loadfile>       # parse @GPU and print the resolution
"""

import argparse
import json
import re
import shutil
import subprocess
import sys

# ====================== CAPABILITY MAP ======================
# Mirrors the per-use-case `validation` map in fold_engines.ENGINE_REGISTRY -- same shape,
# same vocabulary (SUPPORTED / GATED / UNSUPPORTED), so anyone reading the registry sees both
# contracts without being told they exist. When this merges into ENGINE_REGISTRY, these
# become the `splits` key of each engine and this dict goes away.
#
# af2 is UNSUPPORTED for batch DELIBERATELY: AF-2's multi-GPU story is not Lightning device
# sharding, and claiming it works because the flag parses is exactly the thing that bites
# later. If AF-2 batch sharding is wanted, it earns its own status.
SPLITS = {
    "af2":   {"batch": "UNSUPPORTED", "context": "UNSUPPORTED"},
    "boltz": {"batch": "SUPPORTED",   "context": "UNSUPPORTED"},
    # when Fold-CP (arXiv:2603.14806) is installed in its OWN env and gated:
    # "boltz_cp": {"batch": "SUPPORTED", "context": "GATED"},
}

DEFAULTS = {"devices": 1, "split": "batch", "workers": None}
VALID_SPLITS = ("batch", "context")

_GPU_LINE = re.compile(r"^\s*@GPU\b(.*)$", re.M)
_KV = re.compile(r"(\w+)\s*=\s*([^\s]+)")


# ====================== PARSE ======================
def parse_gpu_directive(text):
    """Scan loadfile text for @GPU. Returns the request dict, defaults when absent.

    Omitting @GPU yields devices=1 split=batch, which the SPEC requires to be
    byte-identical to today's behavior -- so `declared` tells a caller whether the
    loadfile actually asked for anything.
    """
    req = dict(DEFAULTS)
    req["declared"] = False
    hits = _GPU_LINE.findall(text or "")
    if not hits:
        return req
    if len(hits) > 1:
        raise SystemExit(
            "@GPU appears %d times; it may appear at most once per loadfile. "
            "One directive answers 'how is this run using the hardware'." % len(hits))
    req["declared"] = True
    # Cut the trailing comment BEFORE scanning for key=value. Without this, `#` is not a
    # comment on the @GPU line at all: `@GPU devices=2 # split=context` silently applied
    # split=context, and `# workers=99 was the old value` silently applied workers=99. A
    # commented-out setting taking effect is the exact failure the rest of this module is
    # built to prevent, so it is a parse-time cut rather than a validation-time complaint.
    # (`# @GPU ...` at line start was already safe -- _GPU_LINE anchors @GPU to the line.)
    spec = hits[0].split("#", 1)[0]
    for key, val in _KV.findall(spec):
        key = key.lower()
        if key not in ("devices", "split", "workers"):
            raise SystemExit(
                "@GPU: unknown key '%s'. Valid keys: devices, split, workers." % key)
        if key in ("devices", "workers"):
            try:
                req[key] = int(val)
            except ValueError:
                raise SystemExit("@GPU %s=%s is not an integer." % (key, val))
        else:
            req[key] = val.lower()
    return req


# ====================== DETECT ======================
def visible_gpu_count():
    """GPUs this box reports, via nvidia-smi. None when that cannot be determined.

    nvidia-smi rather than torch.cuda.device_count() on purpose (SPEC §3b): this must work
    with no model stack loaded. None is distinct from 0 -- "we could not tell" must never be
    reported as "there are none".
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


# ====================== GATE ======================
def validate_gpu_request(engine, devices, split, *, visible=None, executing=True):
    """THE ONLY device-policy gate. Raises SystemExit with an actionable message, or returns.

    Every failure names the offending value, the valid set, and where applicable the engines
    that satisfy it -- so a session that has forgotten all of this can recover the contract
    from one failed run.
    """
    if not isinstance(devices, int) or isinstance(devices, bool) or devices < 1:
        raise SystemExit("@GPU devices=%r is invalid; it must be an integer >= 1." % devices)

    if split not in VALID_SPLITS:
        raise SystemExit("@GPU split=%s is invalid. Valid: %s."
                         % (split, " | ".join(VALID_SPLITS)))

    # context parallelism across one device is a contradiction. Fail, never warn: a silent
    # no-op here produces a run that CLAIMS a capability it did not use (SPEC §3b rule 5).
    if split == "context" and devices == 1:
        raise SystemExit(
            "@GPU split=context with devices=1 is a contradiction -- context parallelism "
            "shards ONE prediction across devices. Use split=batch, or devices>1.")

    caps = SPLITS.get(engine)
    if caps is None:
        raise SystemExit("@GPU: unknown engine '%s'. Known: %s."
                         % (engine, ", ".join(sorted(SPLITS))))

    status = caps.get(split, "UNSUPPORTED")
    if status == "UNSUPPORTED" and devices > 1:
        ok = [e for e, c in sorted(SPLITS.items()) if c.get(split) == "SUPPORTED"]
        raise SystemExit(
            "@GPU split=%s is UNSUPPORTED for engine '%s'. %s"
            % (split, engine,
               ("Engines that support it: %s." % ", ".join(ok)) if ok
               else "No installed engine supports it yet."))
    if status == "GATED" and devices > 1:
        raise SystemExit(
            "@GPU split=%s is GATED for engine '%s' -- plumbed but not validated on this "
            "hardware. It needs an explicit opt-in before it can run." % (split, engine))

    # The device-count rule. SPEC §3b rule 3, enforced only when we are about to fold.
    if executing and devices > 1:
        if visible is None:
            visible = visible_gpu_count()
        if visible is None:
            raise SystemExit(
                "@GPU devices=%d but the visible GPU count could not be determined "
                "(nvidia-smi absent or failed). Refusing to start a multi-device fold "
                "blind. Run with --dry-run to see the plan." % devices)
        if devices > visible:
            raise SystemExit(
                "@GPU devices=%d but only %d GPU%s visible (nvidia-smi). "
                "Lower devices, or run on hardware that has them. "
                "--dry-run will plan for %d on this box without refusing."
                % (devices, visible, " is" if visible == 1 else "s are", devices))


# ====================== RESOLVE ======================
def resolve_request(engine, req, *, executing=True, visible=None):
    """Validate, then report requested vs visible vs used. The provenance record.

    `used` is what will actually be handed to the engine. On a plan-only pass over a box
    with fewer GPUs than asked, requested stays 8 and used drops to what exists -- both are
    kept, because "did this run use multiple GPUs?" must be answerable from the artifacts
    rather than from anyone's recollection (SPEC §3c).
    """
    devices = req.get("devices", 1)
    split = req.get("split", "batch")
    if visible is None:
        visible = visible_gpu_count()

    validate_gpu_request(engine, devices, split, visible=visible, executing=executing)

    if executing:
        used = devices
        note = ("single device (default)" if devices == 1
                else "%d devices, split=%s" % (devices, split))
    else:
        used = devices if (visible or 0) >= devices else max(1, visible or 1)
        if used == devices:
            note = "plan only; this box can satisfy devices=%d" % devices
        else:
            note = ("plan only; requested %d, this box has %s -- would execute on %d here, "
                    "and on %d where %d are present"
                    % (devices, "an undetermined count" if visible is None else str(visible),
                       used, devices, devices))

    return {"engine": engine, "requested": devices, "visible": visible, "used": used,
            "split": split, "workers": req.get("workers"),
            "declared": bool(req.get("declared")), "executing": bool(executing),
            "note": note}


def manifest_fields(res):
    """What goes into the per-run manifest. SPEC §3c -- non-negotiable.

    A pose folded on 4 devices and one folded on 1 must be distinguishable afterwards; if
    they ever disagree numerically, this is how we find out which was which.
    """
    return {"gpu_devices_requested": res["requested"],
            "gpu_devices_used": res["used"],
            "gpu_devices_visible": res["visible"],
            "gpu_split": res["split"],
            "gpu_workers": res["workers"],
            "gpu_declared_in_loadfile": res["declared"]}


def boltz_device_args(res, *, as_requested=False):
    """The argv fragment for `boltz predict`.

    Defaults to `used`, never `requested` -- the command we RUN is the command we print, so a
    log can never claim a device count that did not happen.

    as_requested=True gives the invocation the loadfile asked for, for DISPLAY only: on a
    1-GPU box planning devices=8, that is the line worth showing a reviewer. Printing both
    (see --plan) shows the capability without ever claiming it executed.
    """
    n = res["requested"] if as_requested else res["used"]
    args = ["--devices", str(n)]
    if res.get("workers") is not None:
        args += ["--num_workers", str(res["workers"])]
    return args


def describe(res):
    """One human line for the console and the run log."""
    vis = "unknown" if res["visible"] is None else str(res["visible"])
    return ("@GPU  requested=%d  visible=%s  using=%d  split=%s   %s"
            % (res["requested"], vis, res["used"], res["split"], res["note"]))


# ====================== SELFTEST (no GPU needed) ======================
def selftest():
    fails = []

    def ck(label, cond):
        print("  %-4s %s" % ("ok" if cond else "FAIL", label))
        if not cond:
            fails.append(label)

    def refuses(fn):
        try:
            fn()
            return False
        except SystemExit:
            return True

    print("=" * 74)
    print("  gpu_devices selftest")
    print("=" * 74)

    ck("no @GPU line yields devices=1 split=batch, declared=False",
       parse_gpu_directive(">T\nSEQ\n") == dict(DEFAULTS, declared=False))
    r = parse_gpu_directive("@GPU devices=8 split=batch\n>T\nSEQ\n")
    ck("parses devices=8 split=batch, declared=True",
       r["devices"] == 8 and r["split"] == "batch" and r["declared"])
    ck("explicit devices=1 is identical to omitting the line (minus `declared`)",
       {k: v for k, v in parse_gpu_directive("@GPU devices=1\n").items() if k != "declared"}
       == {k: v for k, v in parse_gpu_directive("").items() if k != "declared"})
    ck("a second @GPU line is refused",
       refuses(lambda: parse_gpu_directive("@GPU devices=2\n@GPU devices=4\n")))
    ck("an unknown key is refused, not ignored",
       refuses(lambda: parse_gpu_directive("@GPU gpus=4\n")))
    ck("a non-integer devices is refused",
       refuses(lambda: parse_gpu_directive("@GPU devices=lots\n")))

    # A trailing comment is a comment, not configuration. Found by probing the parser rather
    # than by reading it: every one of these used to take effect.
    ck("a commented-out key on the @GPU line does NOT take effect",
       parse_gpu_directive("@GPU devices=2  # split=context\n")["split"] == "batch")
    ck("...nor does a commented-out workers value",
       parse_gpu_directive("@GPU devices=8 split=batch  # workers=99 was the old value\n")
       ["workers"] is None)
    ck("the keys BEFORE the comment still apply",
       parse_gpu_directive("@GPU devices=4  # split=context\n")["devices"] == 4)
    ck("an unknown key inside a comment is not an error either",
       parse_gpu_directive("@GPU devices=2  # gpus=4 was wrong\n")["devices"] == 2)
    ck("a fully commented-out directive is not a declaration",
       parse_gpu_directive("# @GPU devices=8\n")["declared"] is False)

    ck("devices=0 is refused",
       refuses(lambda: validate_gpu_request("boltz", 0, "batch", visible=8)))
    ck("an invalid split names the valid set",
       refuses(lambda: validate_gpu_request("boltz", 2, "sideways", visible=8)))
    ck("split=context with devices=1 is REFUSED, not silently ignored",
       refuses(lambda: validate_gpu_request("boltz", 1, "context", visible=8)))
    ck("an unknown engine is refused",
       refuses(lambda: validate_gpu_request("rosetta", 2, "batch", visible=8)))
    ck("context on boltz (UNSUPPORTED) is refused above 1 device",
       refuses(lambda: validate_gpu_request("boltz", 4, "context", visible=8)))
    ck("af2 batch (UNSUPPORTED) is refused above 1 device",
       refuses(lambda: validate_gpu_request("af2", 4, "batch", visible=8)))
    ck("af2 at devices=1 still passes -- today's path is untouched",
       validate_gpu_request("af2", 1, "batch", visible=1) is None)

    ck("EXECUTING: devices=8 on a 1-GPU box is REFUSED",
       refuses(lambda: validate_gpu_request("boltz", 8, "batch", visible=1)))
    ck("EXECUTING: devices=8 on an 8-GPU box passes",
       validate_gpu_request("boltz", 8, "batch", visible=8) is None)
    ck("EXECUTING: devices>1 with an undetermined count is refused, not guessed",
       refuses(lambda: validate_gpu_request("boltz", 2, "batch", visible=None)))

    plan1 = resolve_request("boltz", parse_gpu_directive("@GPU devices=8 split=batch\n"),
                            executing=False, visible=1)
    ck("PLANNING: devices=8 on a 1-GPU box does NOT fail", plan1["requested"] == 8)
    ck("PLANNING: keeps requested=8 while used drops to 1",
       plan1["requested"] == 8 and plan1["used"] == 1)
    ck("PLANNING: the note says what would happen on real hardware",
       "would execute on 1 here" in plan1["note"])

    run8 = resolve_request("boltz", parse_gpu_directive("@GPU devices=8 split=batch\n"),
                           executing=True, visible=8)
    ck("SAME loadfile on an 8-GPU box executes at 8, no flag changed",
       run8["used"] == 8 and run8["requested"] == 8)
    ck("boltz argv carries --devices 8 there",
       boltz_device_args(run8) == ["--devices", "8"])
    ck("boltz argv carries the achievable count on the 1-GPU box",
       boltz_device_args(plan1) == ["--devices", "1"])

    dflt = resolve_request("boltz", parse_gpu_directive(">T\nSEQ\n"),
                           executing=True, visible=1)
    ck("default path is devices=1 -- unchanged behavior when no @GPU is present",
       dflt["used"] == 1 and dflt["requested"] == 1 and not dflt["declared"])

    mf = manifest_fields(run8)
    ck("manifest distinguishes requested from used",
       mf["gpu_devices_requested"] == 8 and mf["gpu_devices_used"] == 8)
    ck("manifest records whether the loadfile asked at all",
       manifest_fields(dflt)["gpu_declared_in_loadfile"] is False)

    ck("workers threads into argv when asked",
       boltz_device_args(resolve_request(
           "boltz", parse_gpu_directive("@GPU devices=8 split=batch workers=4\n"),
           executing=True, visible=8)) == ["--devices", "8", "--num_workers", "4"])

    print()
    if fails:
        print("  %d FAILED" % len(fails))
        for f in fails:
            print("    - %s" % f)
        return 1
    print("  all checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="@GPU device policy: parse, gate, resolve.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true", help="what this box reports")
    ap.add_argument("--plan", metavar="LOADFILE", help="parse @GPU and print the resolution")
    ap.add_argument("--engine", default="boltz")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    if a.show:
        n = visible_gpu_count()
        print("nvidia-smi visible GPUs: %s" % ("undetermined" if n is None else n))
        print("splits capability map:")
        for eng, caps in sorted(SPLITS.items()):
            print("  %-10s %s" % (eng, caps))
        return

    if a.plan:
        req = parse_gpu_directive(open(a.plan, encoding="utf-8").read())
        res = resolve_request(a.engine, req, executing=False)
        if a.json:
            print(json.dumps({"resolution": res, "manifest": manifest_fields(res),
                              "boltz_argv": boltz_device_args(res),
                              "boltz_argv_as_requested":
                                  boltz_device_args(res, as_requested=True)}, indent=2))
        else:
            print(describe(res))
            vis = "undetermined" if res["visible"] is None else res["visible"]
            print("  on this box (%s visible) : boltz predict ... %s"
                  % (vis, " ".join(boltz_device_args(res))))
            if res["requested"] != res["used"]:
                print("  as requested (%d GPUs)   : boltz predict ... %s"
                      % (res["requested"],
                         " ".join(boltz_device_args(res, as_requested=True))))
                print("  ^ the second line is what runs where %d devices exist. Identical "
                      "loadfile, nothing toggled." % res["requested"])
        return

    ap.print_help()


if __name__ == "__main__":
    main()
