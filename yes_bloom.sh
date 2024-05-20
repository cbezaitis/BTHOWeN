#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120000
#SBATCH --job-name="training"
#SBATCH --output=yes_bloom01.out
#SBATCH --mail-user=charalabos.bezaitis@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

source ./venv/bin/activate

python3  ./software_model/train_audio_yes_bloom.py  --filter_inputs 29 --filter_entries 8192 --filter_hashes 4 --bits_per_input 6 --num_workers 1

uname -a