#!/bin/bash
# ============================================================================================
# The first thing we would run on an allocated node.
#
# It answers the one question this repository cannot answer on the hardware it was written on:
# does folding a candidate panel actually get faster as cards are added, and by how much. One
# submission produces the whole table -- 1, 2, 4 and 8 devices over the same fixed panel -- so
# the speedup and the parallel efficiency are measured against each other on identical work.
#
# THIS SCRIPT IS UNTESTED. There is no cluster here; it has never been submitted to a real
# scheduler. It is included because its absence was conspicuous in a compute-allocation
# request and because the shape of the job is itself part of what is being proposed -- not
# because it is known to run. Expect the account, partition and module lines to need editing
# for whatever site this lands on; those are the lines marked EDIT.
#
#   sbatch slurm/scaling.sbatch
#
# ============================================================================================
#SBATCH --job-name=fold-scaling
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1                  # one node: this measures sharding WITHIN a node
#SBATCH --ntasks=1                 # one process, which itself drives all the devices
#SBATCH --gres=gpu:8               # EDIT. Some sites want --gpus=8 or --gpus-per-node=8
#SBATCH --cpus-per-task=16         # see the note on workers below
#SBATCH --mem=96G
#SBATCH --time=02:00:00
##SBATCH --account=EDIT            # EDIT and uncomment
##SBATCH --partition=EDIT          # EDIT and uncomment

set -euo pipefail

# --------------------------------------------------------------------------------------------
# WHY --nodes=1 AND NOT MORE.
#
# split=batch shards an independent candidate list, exchanging nothing between devices, so it
# would distribute across nodes without modification. But a SCALING measurement wants the
# interconnect out of the picture: a 1-to-8 curve inside one node isolates per-device fixed
# costs, which is the thing we actually need to know before asking for more nodes. Multi-node
# comes after this curve exists, not before.
#
# WHY --cpus-per-task=16 FOR 8 GPUS.
#
# The @GPU directive carries `workers=4`, which is the input-loading workers per device. Eight
# devices at four workers is 32 loader threads against 16 cores, which is deliberate
# oversubscription -- the workers are I/O-bound on reading inputs, not compute-bound. If the
# benchmark comes back CPU-starved, this is the first number to raise, and it is also one of
# the things worth learning on real hardware.
# --------------------------------------------------------------------------------------------

module purge 2>/dev/null || true
# EDIT: whatever makes `conda` and the boltz environment available at your site.
# module load cuda/12.4 miniconda
# source activate boltz2

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=============================================================================="
echo " node          : $(hostname)"
echo " job           : ${SLURM_JOB_ID:-none}"
echo " GPUs granted  : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " GPUs on node  : $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
echo "=============================================================================="

# Those last two lines are printed next to each other on purpose, and they are allowed to
# differ. The node's inventory is not the allocation. The device policy reads
# CUDA_VISIBLE_DEVICES first for exactly that reason, so a job granted 2 cards on an 8-card
# node is held to 2 instead of reaching for all 8 -- see allocated_gpu_count() in
# scripts/gpu_devices.py. If these two numbers disagree, the smaller one is the one that
# governs, and a request above it will be refused rather than quietly downgraded.

# What the policy module makes of this machine, recorded before anything is folded. If the job
# fails later, this is the line that says what it thought it had.
python3 scripts/gpu_devices.py --show
python3 scripts/gpu_devices.py --plan example_8gpu.loadfile.txt

# The sweep. --copies is a multiple of the largest device count, or the last device sits idle
# on the final batch and efficiency reads low for a reason that has nothing to do with scaling.
python3 scripts/scaling_benchmark.py \
    --devices 1,2,4,8 \
    --copies 32 \
    --samples 1 \
    --out "scaling_${SLURM_JOB_ID:-local}"

echo
echo "Wrote scaling_${SLURM_JOB_ID:-local}/scaling.json"
echo "Efficiency is speedup / devices. Below roughly 60% at 8 devices, per-device fixed costs"
echo "are dominating and the answer is a larger panel rather than more cards."
