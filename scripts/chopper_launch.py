#!/usr/bin/env python3
"""chopper_launch.py -- the single gate every run goes through. ONE JOB AT A TIME. (curated)

Runs may be started unattended, with one standing condition on
this box: only one job at a time. That condition is not a promise here, it is a LOCK -- an atomic
O_EXCL lockfile holding the pid and the run, checked for staleness by asking the OS whether that pid
is still alive. A second launch does not queue behind a race, it is REFUSED and told what is running.

  chopper_launch.py --run LOADFILE        launch now (refused if the box is busy)
  chopper_launch.py --queue LOADFILE      put it in line; it starts when the box frees up
  chopper_launch.py --pump                start the next queued job if nothing is running
  chopper_launch.py --status              what is running, what is waiting, what finished
  chopper_launch.py --cancel ID           drop a queued job (never touches a running one)
  chopper_launch.py --selftest

WHAT IT DOES NOT DO: it does not score, rank, file, or decide anything. It hands a loadfile to the
console (chopper_vNNN.py --loadfile) and gets out of the way, because the console is where
provenance is earned -- a run folded through here is stamped, and a run folded around it is not.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
try:
    import preflight as pf           # the read-only "what will this loadfile DO" reporter
except Exception:
    pf = None
STATE = ROOT / "state" / "launch"
LOCK = STATE / "RUNNING.lock"
QUEUE = STATE / "queue"
LOGS = STATE / "logs"
HISTORY = STATE / "history.jsonl"


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure():
    for d in (STATE, QUEUE, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def current_chopper(override=None):
    """Which console runs this. Printed every time, never assumed silently -- the version that runs
    is part of the record.

    NOT the highest version number. chopper.py is numerically highest and is PARKED (June, a
    scope that was never merged): it has no --loadfile at all, so picking it dropped a headless
    launch into the interactive menu, where it died on EOF. Two filters instead:

      1. it must actually declare --loadfile (no headless path, not a candidate), and
      2. among those, the most recently COMMITTED wins, because that is what "current" means in
         this repo -- version numbers here record lineage, not recency.
    """
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    only = ROOT / "chopper.py"
    if only.exists():
        return only
    cands = []
    for p in ROOT.glob("chopper.py"):
        tail = p.stem.split("_v")[-1]
        if not tail.isdigit():
            continue
        try:
            if '"--loadfile"' not in p.read_text():
                continue                       # no headless path -- it would sit at a menu forever
        except Exception:
            continue
        when = ""
        try:
            when = subprocess.run(["git", "log", "-1", "--format=%ct", "--", p.name],
                                  cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            pass
        cands.append((int(when) if when.isdigit() else 0, int(tail), p))
    if not cands:
        raise SystemExit("chopper.py not found in %s" % ROOT)
    return max(cands)[2]


# ─────────────────────────────── the lock ───────────────────────────────
def _alive(pid):
    """Is that pid a RUNNING process — not a zombie.

    `os.kill(pid, 0)` succeeds on a zombie, because a finished child keeps its pid entry until its
    parent reaps it. That made the swarm pump deadlock against itself on 2026-10-01: the pump waits
    for the lock to go stale, the lock asked _alive(), _alive() said yes about a `<defunct>` process,
    and the child stayed unreaped precisely BECAUSE the pump was sleeping in that wait loop instead of
    calling wait(). First job finished, folds stuck at 0/6, and an overnight run did nothing at all
    while looking busy — the worst possible failure shape.

    So read the state from /proc and treat Z (zombie) and X (dead) as gone."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            data = fh.read()
        # field 3 is the state; comm (field 2) is parenthesised and may contain spaces
        state = data[data.rindex(b")") + 2:data.rindex(b")") + 3].decode()
        return state not in ("Z", "X", "x")
    except (FileNotFoundError, ProcessLookupError, ValueError, OSError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def read_lock():
    """The running job, or None. A lock whose pid is gone is STALE: reported, cleared, and the job
    it named is recorded as interrupted rather than quietly forgotten."""
    if not LOCK.exists():
        return None
    try:
        info = json.loads(LOCK.read_text())
    except Exception:
        LOCK.unlink(missing_ok=True)
        return None
    if _alive(int(info.get("pid", -1))):
        return info
    info["stale"] = True
    return info


def clear_stale():
    info = read_lock()
    if info and info.get("stale"):
        LOCK.unlink(missing_ok=True)
        info["ended_utc"] = _utc()
        info["outcome"] = "interrupted (the process is gone; the lock was stale)"
        _record(info)
        return info
    return None


def acquire(info):
    """Atomic: O_CREAT|O_EXCL is the whole mutual exclusion. No check-then-write window."""
    _ensure()
    clear_stale()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as fh:
        json.dump(info, fh, indent=2)
    return info


def release(outcome=None):
    info = None
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
        except Exception:
            info = {}
        LOCK.unlink(missing_ok=True)
    if info is not None:
        info["ended_utc"] = _utc()
        if outcome:
            info["outcome"] = outcome
        _record(info)
    return info


def _record(entry):
    _ensure()
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(entry) + "\n")



# ─────────────────────────────── the preflight gate ───────────────────────────────
def preflight_check(loadfile):
    """Run the preflight and let it VETO the launch.

    preflight.py has existed since 2026-08 and checks the traps that are invisible in a loadfile --
    including the record-naming rule that cost us a whole run on 2026-09-18 (a control named for its
    crystal chain, annotations silently dropped, on_target 0 reported for a fold that passed).
    That rule had already been written down three times and fixed twice, in two different producers,
    because nothing made it a GATE. Now it is one: every launch goes through it, and a blocker stops
    the run before the GPU spins.

    It reuses the real parsers, so it cannot drift from the pipeline -- which is the only reason it
    is safe to let it refuse."""
    if pf is None:
        return {"ok": True, "available": False, "blockers": [], "warnings": [],
                "say": "preflight unavailable -- launching unchecked"}
    try:
        rep = pf.analyse(str(loadfile))
    except Exception as e:
        # A loadfile preflight cannot even read is not a loadfile we should hand to the GPU.
        return {"ok": False, "available": True, "blockers": ["preflight could not parse it: %s" % e],
                "warnings": [], "say": "preflight failed to parse the loadfile"}
    blockers = list(rep.get("blockers") or [])
    warnings = list(rep.get("warnings") or []) + list(rep.get("notes") or [])

    # preflight REPORTS the record-naming trap (it prints "<-- WRONG PREFIX" and sets nb=False) but
    # does not call it a blocker -- it still says READY. That is the exact file that cost us a run on
    # 2026-09-18: annotations fail to parse, the co-fold path swallows it, and the report states
    # on_target 0 for a fold that passed. preflight's job is to report facts; deciding what is FATAL
    # is the launcher's. So we promote it here, from preflight's own data, without touching its file.
    for r in (rep.get("records") or []):
        if r.get("role") == "candidate" and not r.get("nb"):
            blockers.append(
                "record '%s' is a candidate but does not start with NB -- loadfile_annotations will "
                "REJECT the file, the co-fold path will swallow that, and the run will score against "
                "an EMPTY goal and report on_target 0. Rename it NB_%s."
                % (r.get("name"), r.get("name")))
        if r.get("role") == "TARGET" and r.get("nb"):
            blockers.append("the TARGET record '%s' starts with NB -- record 1 must not, or the "
                            "target and the binder swap roles." % r.get("name"))
    return {"ok": not blockers, "available": True, "blockers": blockers, "warnings": warnings,
            "mode": rep.get("mode"), "say": ("READY" if not blockers else
                                             "%d blocker(s)" % len(blockers))}


# ─────────────────────────────── launching ───────────────────────────────
def launch(loadfile, *, chopper=None, note="", requested_by="cli", wait=False, extra=(), force=False):
    """Start a run, or refuse: because preflight vetoed it, or because the box is busy."""
    _ensure()
    lf = Path(loadfile)
    if not lf.is_absolute():
        lf = ROOT / lf
    if not lf.is_file():
        raise SystemExit("no such loadfile: %s" % lf)
    pre = preflight_check(lf)
    if not pre["ok"] and not force:
        return {"ok": False, "why": "preflight", "preflight": pre}

    busy = read_lock()
    if busy and not busy.get("stale"):
        return {"ok": False, "why": "busy", "running": busy}

    wp = current_chopper(chopper)
    run_id = "%s_%s" % (lf.stem, datetime.now().strftime("%Y%m%d_%H%M%S"))
    log = LOGS / ("%s.log" % run_id)
    info = {"id": run_id, "loadfile": str(lf), "chopper": wp.name, "note": note,
            "requested_by": requested_by, "started_utc": _utc(), "log": str(log), "pid": None,
            "preflight": {"ok": pre["ok"], "blockers": pre["blockers"],
                          "warnings": pre["warnings"], "forced": bool(force and not pre["ok"])}}
    if acquire(info) is None:
        return {"ok": False, "why": "busy", "running": read_lock()}

    # -u: the child's stdout goes to a FILE here, so Python block-buffers it and the run log sits
    # at 0 bytes for hours. That cost real debugging time twice (the ESMFold triage log on
    # 2026-10-02, the AI briefing before it) because an empty log reads as "nothing happened"
    # rather than "nothing flushed". Unbuffered costs nothing at this cadence.
    cmd = [sys.executable, "-u", str(wp), "--loadfile", str(lf)] + list(extra)
    fh = open(log, "w")
    fh.write("# %s\n# %s\n# requested by: %s\n%s\n\n" % (run_id, _utc(), requested_by, " ".join(cmd)))
    fh.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    info["pid"] = proc.pid
    LOCK.write_text(json.dumps(info, indent=2))
    if not wait:
        return {"ok": True, "launched": info}
    rc = proc.wait()
    release("exit %d" % rc)
    return {"ok": True, "launched": info, "returncode": rc}


# ─────────────────────────────── the queue ───────────────────────────────
def enqueue(loadfile, *, note="", requested_by="cli", priority=50, dedupe=True):
    """Put a loadfile in line. IDEMPOTENT by default.

    The swarm's pump re-plans after every job so dependent arms appear as parents finish -- which
    meant every re-plan re-queued the arms still pending, and the queue grew 3 -> 6 -> 9. The whole
    fold budget would have gone on duplicates of the same three jobs. A loadfile already waiting, or
    already running, is not queued again."""
    _ensure()
    lf = Path(loadfile)
    if not lf.is_absolute():
        lf = ROOT / lf
    if not lf.is_file():
        raise SystemExit("no such loadfile: %s" % lf)
    if dedupe:
        for it in queued():
            if Path(it.get("loadfile", "")).resolve() == lf.resolve():
                return dict(it, duplicate=True)
        cur = read_lock()
        if cur and not cur.get("stale") and cur.get("loadfile"):
            if Path(cur["loadfile"]).resolve() == lf.resolve():
                return {"id": cur.get("id"), "loadfile": str(lf), "duplicate": True,
                        "running": True}
    qid = "%03d_%s_%s" % (priority, datetime.now().strftime("%Y%m%d_%H%M%S"), lf.stem)
    item = {"id": qid, "loadfile": str(lf), "note": note, "requested_by": requested_by,
            "queued_utc": _utc(), "priority": priority}
    (QUEUE / ("%s.json" % qid)).write_text(json.dumps(item, indent=2))
    return item


def queued():
    _ensure()
    out = []
    for p in sorted(QUEUE.glob("*.json")):
        try:
            out.append(json.load(open(p)) | {"_path": str(p)})
        except Exception:
            continue
    return out


def cancel(qid):
    for it in queued():
        if it["id"] == qid or it["id"].endswith(qid):
            Path(it["_path"]).unlink()
            return it
    return None


def pump(chopper=None):
    """Start the next queued job IF the box is free. This is the only thing that ever auto-starts,
    and it starts exactly one."""
    busy = read_lock()
    if busy and not busy.get("stale"):
        return {"ok": False, "why": "busy", "running": busy}
    q = queued()
    if not q:
        return {"ok": False, "why": "empty"}
    nxt = q[0]
    res = launch(nxt["loadfile"], chopper=chopper, note=nxt.get("note", ""),
                 requested_by=nxt.get("requested_by", "queue"))
    if res.get("ok"):
        Path(nxt["_path"]).unlink(missing_ok=True)
    return res


def history(n=10):
    if not HISTORY.exists():
        return []
    lines = HISTORY.read_text().strip().splitlines()[-n:]
    return [json.loads(l) for l in lines if l.strip()]


def status():
    _ensure()
    # Reporting what is running must not depend on the console being installed. It used to:
    # current_chopper() raises when it finds none, so `--status` died with "not found"
    # instead of answering the question asked, which is whether the box is busy.
    try:
        console = current_chopper().name
    except SystemExit:
        console = None
    return {"running": read_lock(), "queued": queued(), "recent": history(6),
            "chopper": console}


def show_status():
    st = status()
    print("=" * 78)
    print("  CHOPPER LAUNCH -- one job at a time, enforced by %s" % LOCK.name)
    print("=" * 78)
    r = st["running"]
    if r and not r.get("stale"):
        started = r.get("started_utc", "?")
        print("  RUNNING   %s" % r["id"])
        print("            pid %s since %s via %s" % (r.get("pid"), started, r.get("chopper")))
        print("            %s" % r["loadfile"])
        print("            log: %s" % r.get("log"))
    elif r and r.get("stale"):
        print("  STALE     %s -- the process is gone. Clearing on the next launch." % r["id"])
    else:
        print("  RUNNING   nothing. The box is free%s."
              % ("" if st["chopper"] is None else
                 " (console version: %s)" % st["chopper"]))
    print()
    if st["queued"]:
        print("  WAITING   %d" % len(st["queued"]))
        for it in st["queued"]:
            print("    %-46s %s" % (it["id"][:46], Path(it["loadfile"]).name))
    else:
        print("  WAITING   nothing queued.")
    if st["recent"]:
        print()
        print("  RECENT")
        for h in reversed(st["recent"]):
            print("    %-30s %-22s %s" % (h.get("id", "?")[:30], h.get("ended_utc", "?"),
                                          h.get("outcome", "")))


def selftest():
    import tempfile
    fails = []

    def ck(label, cond):
        print("  %-4s %s" % ("ok" if cond else "FAIL", label))
        if not cond:
            fails.append(label)

    global STATE, LOCK, QUEUE, LOGS, HISTORY
    keep = (STATE, LOCK, QUEUE, LOGS, HISTORY)
    print("=" * 78)
    print("  chopper_launch selftest")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as tmp:
        STATE = Path(tmp) / "launch"
        LOCK, QUEUE, LOGS, HISTORY = STATE / "RUNNING.lock", STATE / "queue", STATE / "logs", STATE / "history.jsonl"
        _ensure()
        ck("nothing running on a clean box", read_lock() is None)
        a = acquire({"id": "job-a", "pid": os.getpid(), "loadfile": "x"})
        ck("first job takes the lock", a is not None)
        b = acquire({"id": "job-b", "pid": os.getpid(), "loadfile": "y"})
        ck("SECOND JOB IS REFUSED while the first holds the lock", b is None)
        ck("the lock names what is running", read_lock()["id"] == "job-a")
        release("done")
        ck("lock released", read_lock() is None)
        ck("the finished job is on the record", history(1)[0]["id"] == "job-a")

        # a dead pid must not block the box forever
        LOCK.write_text(json.dumps({"id": "ghost", "pid": 999999, "loadfile": "z"}))
        ck("a dead job's lock reads as stale", (read_lock() or {}).get("stale") is True)
        ck("stale lock is cleared, not inherited", acquire({"id": "job-c", "pid": os.getpid()}) is not None)
        ck("the interrupted job is recorded, not forgotten",
           any(h.get("id") == "ghost" for h in history(5)))
        release("done")

        lf = Path(tmp) / "t.loadfile.txt"
        lf.write_text("@COFOLD samples=3\n")
        it = enqueue(lf, note="test", priority=10)
        ck("queueing writes a waiting job", len(queued()) == 1 and queued()[0]["id"] == it["id"])
        again = enqueue(lf, note="test", priority=10)
        ck("queueing the SAME loadfile twice does not duplicate it",
           len(queued()) == 1 and again.get("duplicate") is True)
        acquire({"id": "running-one", "pid": os.getpid(), "loadfile": str(lf)})
        ck("a loadfile already RUNNING is not queued again",
           enqueue(lf).get("duplicate") is True and len(queued()) == 1)
        release("done")
        # a SECOND, distinct loadfile -- the old test queued the same file twice, which the dedupe
        # above now (correctly) collapses
        lf2 = Path(tmp) / "t2.loadfile.txt"
        lf2.write_text("@COFOLD samples=3\n")
        hi = enqueue(lf2, note="later", priority=90)
        ck("priority orders the queue", queued()[0]["priority"] == 10)
        ck("cancel drops a waiting job", cancel(hi["id"]) is not None and len(queued()) == 1)
        acquire({"id": "blocker", "pid": os.getpid()})
        ck("pump refuses to start a second job while one runs", pump().get("why") == "busy")
        release("done")
        ck("queue survives a busy pump", len(queued()) == 1)

        # The console is a SEPARATE install. current_chopper() raises when none is present,
        # which used to abort the whole suite -- a launcher's own tests failing because the
        # thing it launches is not installed. That is a missing dependency, not a defect in
        # the lock, so it is reported as skipped and the lock checks above still stand.
        try:
            wp = current_chopper()
        except SystemExit as exc:
            wp = None
            print("  skip  console checks -- %s" % exc)
        if wp is not None:
            ck("the console picked has a headless path", '"--loadfile"' in wp.read_text())

        # the gate that would have caught 2026-09-18: a binder record not named NB*
        bad = Path(tmp) / "bad.cofold.txt"
        bad.write_text("@COFOLD samples=3\n@EPITOPE Site = 1-5\n"
                       "@CHAIN 5E94_B msa=server\n@CHAIN T msa=server\n"
                       ">T\nACDEFGHIKLMNPQRSTVWY\n>5E94_B\nQVQLVESGGGQVQLVESGGG\n")
        pc = preflight_check(bad)
        if pc["available"]:
            ck("preflight sees a non-NB binder record as a blocker", not pc["ok"])
            r = launch(bad)
            ck("a blocked loadfile is REFUSED before the lock is taken",
               (not r.get("ok")) and r.get("why") == "preflight")
            ck("refusing on preflight leaves the box free", read_lock() is None)
            r2 = launch(bad, force=True)
            ck("--force overrides, and the override is recorded",
               r2.get("ok") and r2["launched"]["preflight"]["forced"] is True)
            release("done")
        else:
            # Distinguish "not installed here" from "installed and broken". A separate
            # component being absent is a missing dependency, and the lock checks above are
            # unaffected by it; a preflight that is PRESENT and will not import is a real
            # failure, because every launch is supposed to go through it.
            if (HERE / "preflight.py").exists():
                ck("preflight is present but did not import -- the launch gate is DOWN", False)
            else:
                print("  skip  preflight checks -- preflight.py is not installed here")

        try:
            launch(Path(tmp) / "nope.txt")
            ck("a missing loadfile is refused", False)
        except SystemExit:
            ck("a missing loadfile is refused", True)
    STATE, LOCK, QUEUE, LOGS, HISTORY = keep
    print()
    if fails:
        print("  %d FAILED: %s" % (len(fails), "; ".join(fails)))
        return 1
    print("  all checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Launch chopper runs, one at a time.")
    ap.add_argument("--run", metavar="LOADFILE")
    ap.add_argument("--queue", metavar="LOADFILE")
    ap.add_argument("--pump", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--cancel", metavar="ID")
    ap.add_argument("--note", default="")
    ap.add_argument("--by", default="cli", help="who asked for this run (agent, console, cli)")
    ap.add_argument("--priority", type=int, default=50)
    ap.add_argument("--chopper", help="console version to use (default: newest on disk)")
    ap.add_argument("--wait", action="store_true", help="block until it finishes")
    ap.add_argument("--force", action="store_true",
                    help="launch even if preflight found a blocker (recorded as forced)")
    ap.add_argument("--resume-dock", action="store_true",
                    help="pass --resume-dock to the console: dock the working loadfile already in "
                         "the run dir instead of regenerating it. For picking up a run whose dock "
                         "stage died after generation and triage had finished.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.status:
        print(json.dumps(status(), indent=2)) if a.json else show_status()
        return 0
    if a.cancel:
        it = cancel(a.cancel)
        print("cancelled %s" % it["id"] if it else "no such queued job: %s" % a.cancel)
        return 0 if it else 1
    if a.queue:
        it = enqueue(a.queue, note=a.note, requested_by=a.by, priority=a.priority)
        print("queued %s  (%d waiting)" % (it["id"], len(queued())))
        return 0
    if a.pump:
        res = pump(a.chopper)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if a.run:
        res = launch(a.run, chopper=a.chopper, note=a.note, requested_by=a.by, wait=a.wait,
                     force=a.force, extra=(("--resume-dock",) if a.resume_dock else ()))
        if not res.get("ok") and res.get("why") == "preflight":
            pre = res["preflight"]
            print("REFUSED by preflight -- %s" % pre["say"])
            for b in pre["blockers"]:
                print("  ✖ %s" % b)
            print("  Fix the loadfile, or --force to launch anyway (it is recorded as forced).")
            return 1
        if not res.get("ok"):
            r = res.get("running") or {}
            print("REFUSED -- the box is busy with %s (pid %s, since %s)."
                  % (r.get("id"), r.get("pid"), r.get("started_utc")))
            print("  queue it instead:  chopper_launch.py --queue %s" % a.run)
            return 1
        i = res["launched"]
        print("launched %s  pid %s  via %s" % (i["id"], i["pid"], i["chopper"]))
        print("  log: %s" % i["log"])
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
