#!/bin/bash
# ============================================================================================
# The production shape: one design campaign, eight devices, task-parallel.
#
# REQUIRES THE DESIGN PIPELINE, WHICH IS NOT IN THIS REPOSITORY. This script will not run here
# and is not meant to -- `chopper_launch.py --run` needs the console and the fold stack, and
# neither is public. It is included because "how would you actually use the allocation" is a
# fair question about a compute request, and the answer is a job script rather than a
# paragraph. Everything it depends on that IS here -- the device policy, the lock, the queue --
# is the part a reviewer can check.
#
# ALSO UNTESTED against a real scheduler, for the same reason as slurm/scaling.sbatch.
#
#   sbatch slurm/campaign.sbatch my_panel.loadfile.txt
#
# ============================================================================================
#SBATCH --job-name=fold-campaign
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8               # EDIT for the site's syntax
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
##SBATCH --account=EDIT
##SBATCH --partition=EDIT

set -euo pipefail

LOADFILE="${1:?usage: sbatch slurm/campaign.sbatch <loadfile>}"

module purge 2>/dev/null || true
# EDIT: site modules, as in slurm/scaling.sbatch.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "loadfile      : ${LOADFILE}"
echo "GPUs granted  : ${CUDA_VISIBLE_DEVICES:-<unset>}"

# --------------------------------------------------------------------------------------------
# THE DEVICE COUNT IS NOT IN THIS FILE, AND THAT IS THE POINT.
#
# It is in the loadfile, on its @GPU line, next to the science it belongs to. This script asks
# SLURM for hardware; the loadfile declares what the run intends to use. Two consequences
# worth stating:
#
#   1. The same loadfile runs unchanged on this node and on the workstation. On eight cards it
#      executes at eight; on one card, planning still resolves and printing still works. No
#      flag is toggled between the two, which is what stops a demonstration from being a
#      special case.
#
#   2. If --gres here and @GPU there disagree, the run is REFUSED and both numbers are named.
#      That is deliberate: a mismatch is a mistake in the submission, and the failure mode it
#      replaces -- running at one-eighth of the requested width and reporting success -- is the
#      one that quietly wastes an allocation. Resolve it by editing one of the two numbers, not
#      by adding an override, because there isn't one.
#
# Check the pairing before the queue ever starts, so a mismatch costs a second rather than a
# node-hour:
#
#   python3 scripts/gpu_devices.py --plan my_panel.loadfile.txt
# --------------------------------------------------------------------------------------------

python3 scripts/gpu_devices.py --plan "${LOADFILE}"

# --wait makes the batch job's lifetime the run's lifetime, so SLURM's accounting matches the
# work and the job does not exit while the fold is still going.
#
# The launcher's one-job-at-a-time lock is not redundant with SLURM. SLURM stops two JOBS from
# being given the same cards; the lock stops two RUNS inside one allocation from overlapping --
# the case where a campaign is still folding and something else is told to start. It is an
# O_EXCL lockfile holding a pid, and a crashed run is detected as stale and cleared rather than
# wedging the node for the rest of the wall time. On a shared machine that is the difference
# between a dead job at 2am and a dead job plus seven idle hours.
python3 scripts/chopper_launch.py --run "${LOADFILE}" --by cli --wait

echo
python3 scripts/chopper_launch.py --status
