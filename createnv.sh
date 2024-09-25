#!/bin/sh
echo "Load python"
module load python/3.10.8
echo "Create and activate python venv"
if [ -d "./mypython" ]; then
  echo "Venv already exist. Skipping step"
else
  echo "Creating venv"
  python3 -m venv ./mypython
fi
. ./mypython/bin/activate
echo "Install requirements"
pip install -r requirements.txt
echo "Send the job to execute"
sbatch job.sh
wait 1
echo "To check the job status: watch -n 1 squeue"
