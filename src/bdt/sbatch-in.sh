#!/bin/bash
#SBATCH --nodes=1                       # number of nodes requested
#SBATCH --ntasks=40                     # number of tasks (default: 1)
#SBATCH --ntasks-per-node=40            # number of tasks per node (default: whole node)
#SBATCH --partition=maxwell             # partition to run in (all or maxwell)
#SBATCH --job-name=BDT-UUID             # job name
#SBATCH --output=BDT-UUID-%N-%j.out     # output file name
#SBATCH --error=BDT-UUID-%N-%j.err      # error file name
#SBATCH --time=96:00:00                 # runtime requested
#SBATCH --mail-user=ayan.paul@desy.de   # notification email
#SBATCH --mail-type=END,FAIL            # notification type
#SBATCH --export=ALL
export LD_PRELOAD=""

# load module
module load python/3.10

# run the application:
source .venv/bin/activate
python3.10 bdt.py config-UUID.json
deactivate
