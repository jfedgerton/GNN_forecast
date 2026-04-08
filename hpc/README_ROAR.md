# GNN Forecast on ROAR Collab

## Quick Start

```bash
# 1. SSH in (or use the OOD shell at portal.hpc.psu.edu)
ssh jfe4@submit.hpc.psu.edu

# 2. Clone the repo into scratch
cd /scratch/jfe4
git clone <your-repo-url> GNN_forecast
cd GNN_forecast

# 3. Run one-time environment setup (login node is fine)
bash hpc/setup_env.sh

# 4. Submit jobs
sbatch hpc/job_export_data.sh        # R data export (CPU)
sbatch --dependency=afterok:<JOBID> hpc/job_build_layers.sh   # build layers (CPU)
sbatch --dependency=afterok:<JOBID> hpc/job_train_gpu.sh      # train + forecast (GPU)

# Or run the full master pipeline on GPU:
sbatch hpc/job_master.sh
```

## File Layout on ROAR

```
/scratch/jfe4/GNN_forecast/          # project root
├── hpc/                             # <-- these scripts
│   ├── setup_env.sh                 # one-time env setup
│   ├── activate.sh                  # source this each session
│   ├── job_export_data.sh           # SLURM: R data export
│   ├── job_build_layers.sh          # SLURM: build network layers
│   ├── job_train_gpu.sh             # SLURM: train + forecast (GPU)
│   └── job_master.sh               # SLURM: full pipeline (GPU)
├── .venv/                           # Python virtual environment
├── data/processed/                  # peacesciencer CSV exports
├── data/model_inputs/               # tensors for training
├── outputs/                         # forecasts + intervention CSVs
└── logs/                            # SLURM job logs
```

## Storage Paths

| Path | Quota | Use for |
|------|-------|---------|
| `/storage/home/jfe4` | 16 GB | dotfiles, small configs |
| `/storage/work/jfe4` | 128 GB | persistent data/results you want to keep |
| `/scratch/jfe4` | 50 TB | active project, venv, intermediate outputs |

Scratch is **not backed up** and files are purged after ~90 days of inactivity.
Copy important results to `/storage/work/jfe4` when done.

## Account & Partition Info

- **Credit account:** `jfe4_cr_default` (~517 credits as of Apr 2026)
- **GPU jobs:** use `--partition=sla-prio --gpus=1` (A100 or P100 available)
- **CPU jobs:** use `--partition=standard`
- **Open partition:** `--account=open` (limited, free, good for quick tests)

## Day-to-Day Usage

```bash
# Activate environment
source /scratch/jfe4/GNN_forecast/hpc/activate.sh

# Interactive GPU session (for debugging)
salloc --account=jfe4_cr_default --partition=sla-prio --gpus=1 --mem=32G --time=02:00:00
source hpc/activate.sh
python scripts/master.py --data-dir tests/fixtures/tiny_processed --summary

# Check job status
squeue -u jfe4

# View job output
tail -f logs/train_<JOBID>.out
```

## Module Versions

The setup script loads these modules (adjust versions after running `module spider`):

- `python/3.11.2`
- `cuda/12.6.0`
- `r/4.3.1`

If exact versions differ on RC, run `module spider python` etc. and update
`setup_env.sh` and `activate.sh` accordingly.

## Chaining Jobs with Dependencies

```bash
JOB1=$(sbatch --parsable hpc/job_export_data.sh)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} hpc/job_build_layers.sh)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} hpc/job_train_gpu.sh)
echo "Submitted chain: ${JOB1} → ${JOB2} → ${JOB3}"
```
