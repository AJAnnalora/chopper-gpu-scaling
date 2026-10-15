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
import os
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
def physical_gpus():
    """(count, {uuids}) for the cards in this box, or None if that cannot be established.

    Separate from visible_gpu_count() because CUDA_VISIBLE_DEVICES cannot be validated without
    it: "is index 9 a real device" and "is this UUID one of ours" are questions only the
    hardware can answer, and a token that is not a real device changes how CUDA reads the rest
    of the list. One nvidia-smi call answers both.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    uuids = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
    return len(uuids), uuids


def allocated_gpu_count(physical=None):
    """GPUs THIS PROCESS may use, from the scheduler's own variable. None if it is unset.

    ALSO None when the variable is set but cannot be validated, because the only honest
    answer then is "I do not know" -- see the note on the rules below.

    THE RULES ARE CUDA'S, AND THEY ARE NOT "COUNT THE COMMAS". Verified against
    torch.cuda.device_count() on this box rather than read off the documentation, because the
    documented rule covers only the first of these:

        CUDA_VISIBLE_DEVICES    devices CUDA actually exposes
        0                       1
        0,9                     1    index 9 is not a device -> the list TRUNCATES there
        9                       0    truncated at the first token, so nothing survives
        1,0                     0    same, on a one-card box: index 1 is already invalid
        abc,0                   0    any unparsable token truncates
        0,abc                   1
        0,                      1    trailing and empty tokens are ignored
        00                      1    parsed as an integer, so leading zeros are fine
        0,0                     0    a DUPLICATE rejects the WHOLE list. Not a truncation
        0,<uuid of device 0>    1    index and UUID forms do not mix; the second truncates
        <uuid>                  1    a real UUID resolves
        GPU-nonsense            0
        -1                      0    the documented idiom for "no devices"

    The duplicate case is the one that makes counting tokens actively dangerous rather than
    merely imprecise. `0,0` LOOKS like two devices and is zero, so a stale or hand-edited
    allocation string would have opened the gate on a number the job does not own -- the exact
    class of failure the scheduler fix was written to close, reintroduced one layer down.

    Returning None when the variable is set but the hardware cannot be enumerated is
    deliberate. Without the physical device list, "is index 9 valid" is unanswerable, so the
    count is a guess; this module's standing rule is to refuse rather than guess, and
    multi-device EXECUTION is what gets refused. Planning still resolves anywhere.

    WHY THIS EXISTS, AND WHY IT COMES FIRST. nvidia-smi reports the GPUs PHYSICALLY IN THE
    BOX. On a shared cluster node those are not the GPUs you were given. SLURM hands you a
    subset and says so in CUDA_VISIBLE_DEVICES: allocated 2 of 8 means CUDA_VISIBLE_DEVICES=3,5
    while nvidia-smi still truthfully answers 8.

    Trusting nvidia-smi there defeats the entire gate. `devices=8` would pass the execute
    check (8 <= 8), the fold engine would be told to use 8, and the run would either crash or
    reach for cards belonging to another job -- the exact "quietly ran at the wrong device
    count" failure this module exists to refuse. It cannot be caught on a single-GPU
    workstation, because a one-card box has no subset to be given.

    An empty string is CUDA's way of saying "no devices", so it means 0 and not "unknown" --
    the caller must be able to tell a deliberate zero from a failed detection. Entries may be
    indices or GPU-UUID strings; both are counted the same way. A "-1" entry disables CUDA, so
    anything that is neither an index nor a UUID is treated as the end of the usable list,
    which is what CUDA itself does.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return 0

    # The hardware is consulted LAZILY, only once a token is reached that cannot be judged
    # without it. This is not an optimisation: "-1" and "abc" are invalid by their FORM, so
    # they expose no devices whatever hardware is present, and a machine with no GPU at all
    # must still be able to resolve them. Demanding the device list up front made the whole
    # self-test fail on a GPU-less box -- which is precisely the machine a reviewer uses.
    count, uuids = None, None
    looked_up = False

    def hardware():
        nonlocal count, uuids, looked_up
        if not looked_up:
            looked_up = True
            got = physical if physical is not None else physical_gpus()
            if got is not None:
                count, uuids = got
        return count is not None

    accepted = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            break              # trailing or empty token: the list ends here
        if part.startswith("MIG-"):
            # Addressed by MIG-<uuid>, which --query-gpu=uuid does not list, so this cannot
            # confirm it. Guessing would be the same mistake in a smaller font.
            return None
        if part.isdigit():
            if not hardware():
                return None    # needs the device count, and there isn't one
            ok = int(part) < count
        elif part.startswith("GPU-"):
            if not hardware():
                return None    # needs the real UUIDs
            ok = part in uuids
        else:
            ok = False         # "-1", "abc", anything else: invalid by form, no lookup needed
        if not ok:
            break              # truncate at the first token that is not a real device
        if part in accepted:
            return 0           # a duplicate rejects the whole list, not just the remainder
        accepted.append(part)

    # Index and UUID forms do not mix: once the list is one kind, a token of the other kind
    # truncates it. Enforced here rather than in the loop so the duplicate rule still sees the
    # full prefix it needs.
    if accepted:
        kind = accepted[0].isdigit()
        for i, part in enumerate(accepted):
            if part.isdigit() != kind:
                return i
    return len(accepted)


def visible_gpu_count():
    """GPUs this process may use. The scheduler's allocation first, else what the box reports.

    None when that cannot be determined. None is distinct from 0 -- "we could not tell" must
    never be reported as "there are none".

    nvidia-smi rather than torch.cuda.device_count() for the fallback, on purpose (SPEC §3b):
    this must work with no model stack loaded. But nvidia-smi answers a different question than
    the one being asked -- see allocated_gpu_count(), which is consulted first precisely
    because on a shared node the two answers differ and only one of them is ours to use.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        # Set, so it governs -- including when it resolves to 0, and including when it cannot
        # be validated and the answer is None. Falling back to the box's inventory here would
        # reinstate the original bug: an unverifiable allocation would be silently replaced by
        # a number the job may not own.
        return allocated_gpu_count()
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
            # Name WHERE the number came from. "only 2 visible" is baffling on a node with
            # eight cards in it until you know the scheduler is being quoted, and that is
            # exactly the case where the message has to be self-explanatory -- the reader is
            # looking at a failed cluster job and `nvidia-smi` saying eight.
            # Discriminate on whether the variable is SET, not on whether it could be
            # validated. Asking allocated_gpu_count() again re-derives what the caller has
            # already settled, and gets it wrong wherever the allocation is unverifiable --
            # blaming nvidia-smi for a number that came from the scheduler.
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                src = ("allocated to this job by the scheduler; CUDA_VISIBLE_DEVICES=%s"
                       % os.environ.get("CUDA_VISIBLE_DEVICES", ""))
                how = ("Ask the scheduler for %d (e.g. --gres=gpu:%d), or lower devices to "
                       "match the allocation." % (devices, devices))
            else:
                src = "present on this box, per nvidia-smi"
                how = "Lower devices, or run on hardware that has them."
            raise SystemExit(
                "@GPU devices=%d but only %d GPU%s %s. %s "
                "--dry-run will plan for %d anywhere without refusing."
                % (devices, visible, " is" if visible == 1 else "s are", src, how, devices))


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

    # ================== THE SCHEDULER'S ALLOCATION ==================
    # These are the cluster cases: nvidia-smi is right about the hardware and wrong about what
    # this job was given. A single-GPU workstation cannot produce them by accident, so an
    # eight-card node is simulated by passing `physical` rather than by trusting this box.
    #
    # EVERY EXPECTED VALUE BELOW WAS VERIFIED AGAINST torch.cuda.device_count() on real
    # hardware before being written down. They are not read off NVIDIA's documentation, which
    # describes the truncation rule and says nothing about duplicates -- and duplicates turn
    # out to be the dangerous case.
    EIGHT = (8, {"GPU-%d" % i for i in range(8)})
    _saved = os.environ.get("CUDA_VISIBLE_DEVICES")

    def alloc(value):
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        return allocated_gpu_count(physical=EIGHT)

    try:
        ck("a 2-GPU allocation on an 8-GPU node reports 2, not 8", alloc("3,5") == 2)
        ck("...and devices=2, which is what was granted, passes",
           validate_gpu_request("boltz", 2, "batch", visible=alloc("3,5")) is None)
        ck("...while devices=8 in that allocation is REFUSED rather than stealing cards",
           refuses(lambda: validate_gpu_request("boltz", 8, "batch", visible=alloc("3,5"))))
        os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
        try:
            validate_gpu_request("boltz", 8, "batch", visible=2)
            msg = ""
        except SystemExit as e:
            msg = str(e)
        # The refusal has to say WHERE the count came from, or it reads as nonsense next to an
        # nvidia-smi reporting eight cards on the same node.
        ck("...and the refusal blames the scheduler, not nvidia-smi",
           "scheduler" in msg and "CUDA_VISIBLE_DEVICES=3,5" in msg and "nvidia-smi" not in msg)
        ck("...and says how to fix it from the submission side", "--gres=gpu:8" in msg)

        # Tokens are not devices. Counting commas gets every one of these wrong.
        ck("an out-of-range index TRUNCATES the list, as it does for CUDA",
           alloc("3,99") == 1)
        ck("...and an out-of-range index FIRST leaves nothing", alloc("99,3") == 0)
        ck("an unparsable token truncates", alloc("3,wat") == 1)
        ck("...and truncates from the front too", alloc("wat,3") == 0)
        ck("a trailing comma is not a device", alloc("3,") == 1)
        ck("leading zeros still parse as an index", alloc("03") == 1)
        ck("surrounding whitespace is tolerated", alloc("  3 , 5 ") == 2)

        # THE DANGEROUS ONE. `0,0` looks like two devices and CUDA exposes NONE: a duplicate
        # rejects the whole list rather than truncating it. Counting tokens would have
        # answered 2 and opened the gate on a card the job does not own twice over -- the same
        # failure class as reading nvidia-smi instead of the allocation, one layer down.
        ck("a DUPLICATE index rejects the whole list, which is 0 and not 1", alloc("3,3") == 0)
        ck("...including with whitespace between them", alloc(" 3 ,  3 ") == 0)
        ck("...and three of them", alloc("3,3,3") == 0)
        ck("...and duplicate UUIDs likewise", alloc("GPU-1,GPU-1") == 0)

        ck("a UUID allocation resolves against the real UUIDs", alloc("GPU-1,GPU-2") == 2)
        ck("an unknown UUID is not a device", alloc("GPU-nope") == 0)
        ck("index and UUID forms do not mix; the odd one out truncates",
           alloc("3,GPU-1") == 1)
        ck("an empty allocation is 0 -- a deliberate zero, distinct from undetermined",
           alloc("") == 0)
        ck("the CUDA disable idiom -1 yields 0 usable devices", alloc("-1") == 0)

        ck("MIG instances, which cannot be confirmed, are undetermined rather than guessed",
           alloc("MIG-abc,MIG-def") is None)

        # Unverifiable is not the same as unset, and must not fall back to the inventory.
        # physical_gpus() is stubbed out because "nvidia-smi is missing" cannot be arranged
        # any other way on a box that has it, and asserting this only where it happens to be
        # absent would be a test that silently never runs.
        _real_physical = globals()["physical_gpus"]
        try:
            globals()["physical_gpus"] = lambda: None
            os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
            ck("an allocation that cannot be validated is None -- refuse rather than guess",
               allocated_gpu_count() is None)
            ck("...and visible_gpu_count does NOT fall back to the box's inventory there",
               visible_gpu_count() is None)
            # Hardware-independent answers must survive on a box with no GPU, because that
            # is the machine a reviewer runs this on. Asking for the device list before
            # judging a token that needs no device list made this whole suite fail there.
            for value, want in (("", 0), ("-1", 0), ("abc", 0), ("wat,0", 0)):
                os.environ["CUDA_VISIBLE_DEVICES"] = value
                ck("...while %r resolves to %d with no hardware to consult"
                   % (value, want), allocated_gpu_count() == want)
            # The refusal must still credit the scheduler here. It decides that from whether
            # the variable is SET, not from whether the count could be validated -- asking
            # again would re-derive what the caller already settled and get it backwards in
            # exactly this case.
            os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
            try:
                validate_gpu_request("boltz", 8, "batch", visible=2)
                m2 = ""
            except SystemExit as e:
                m2 = str(e)
            ck("...and an unverifiable allocation STILL blames the scheduler, not nvidia-smi",
               "scheduler" in m2 and "nvidia-smi" not in m2)
        finally:
            globals()["physical_gpus"] = _real_physical

        # An allocation of zero is a real answer and must survive, or a job granted nothing
        # would quietly run on the whole box.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        ck("an allocation of 0 is NOT replaced by the box's inventory",
           visible_gpu_count() == 0)
        ck("...and a multi-device request against it is refused",
           refuses(lambda: validate_gpu_request("boltz", 2, "batch",
                                                visible=visible_gpu_count())))

        del os.environ["CUDA_VISIBLE_DEVICES"]
        ck("unset falls back to the box's own count, which is how a workstation behaves",
           allocated_gpu_count() is None)
    finally:
        if _saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _saved

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
        # Print the two numbers separately and say which governs. The old single line was
        # labelled "nvidia-smi visible GPUs" whatever the source, so under an allocation it
        # reported the allocation and credited nvidia-smi -- "nvidia-smi visible GPUs: 2" on a
        # node where nvidia-smi says 8. That is the baffling message the refusal text was
        # fixed to avoid, still being printed one command over.
        n = visible_gpu_count()
        phys = physical_gpus()
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        print("GPUs in this box (nvidia-smi) : %s"
              % ("undetermined" if phys is None else phys[0]))
        if cvd is None:
            print("CUDA_VISIBLE_DEVICES          : <unset>  (no allocation; the box governs)")
        else:
            print("CUDA_VISIBLE_DEVICES          : %r" % cvd)
        print("usable by this process        : %s%s"
              % ("undetermined" if n is None else n,
                 "   <- the allocation governs" if cvd is not None else ""))
        if cvd is not None and phys is not None and n is not None and n != len(cvd.split(",")):
            # Say so out loud: the difference between tokens and devices is the whole lesson
            # of this variable, and a silent discrepancy here is what sent us looking.
            print("  note: %d comma-separated token(s) resolve to %d real device(s) -- CUDA "
                  "truncates at the first\n        entry that is not a device, and rejects "
                  "the whole list if one repeats." % (len(cvd.split(",")), n))
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
