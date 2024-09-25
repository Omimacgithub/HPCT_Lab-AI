#!/bin/sh
echo "Load python"
module load python cesga/system python/3.10.8
echo "Create and activate python venv"
python3 -m venv ./mypython
. ./mypython/bin/activate
echo "Install requirements"
pip install -r requirements.txt
echo "Send the job to execute"
sbatch job.sh
