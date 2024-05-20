#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120000
#SBATCH --job-name="synthesis"
#SBATCH --output=workspace00.out
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

# CLUSTER_XILINX_DIR=${CLUSTER_XILINX_DIR:=/cluster/projects/itea_lille-ie-idi/opt/Xilinx}
# XILINXD_LICENSE_FILE=27000@xilinx.lisens.ntnu.no
source ../venv/bin/activate


# Following command for letter works
# make template MODEL=../software_model/selected_models/letter.pickle.lzma HASH_UNITS=2 BUS_WIDTH=32

make template MODEL=../software_model/selected_models/mnist_large.pickle.lzma BUS_WIDTH=512
echo "mnist_large.pickle.lzma BUS_WIDTH=512 with intermediate buffer"
# export XILINXD_LICENSE_FILE=27000@xilinx.lisens.ntnu.no
# export LM_LICENSE_FILE=27000@xilinx.lisens.ntnu.no
export LD_LIBRARY_PATH=/cluster/home/charalab/vivado:$LD_LIBRARY_PATH
source /cluster/projects/itea_lille-ie-idi/env/vivado.sh
# source /cluster/projects/itea_lille-ie-idi/env/vivado.sh
# ./cluster/projects/itea_lille-ie-idi/env/vivado.sh
# source /cluster/projects/itea_lille-ie-idi/env/xilinx.sh
# source /cluster/projects/itea_lille-ie-idi/env/xrt.sh
# source /cluster/projects/itea_lille-ie-idi/env/fpga-util.sh
# export LD_LIBRARY_PATH=/cluster/home/charalab/vivado:$LD_LIBRARY_PATH
# export XILINXD_LICENSE_FILE=27000@xilinx.lisens.ntnu.no
# cd ./working_example
vivado -mode tcl -source flow.tcl 

# Remove Vivado logs
rm *.jou
rm *.log

uname -a