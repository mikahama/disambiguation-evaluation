#!/bin/bash -l
# created: Nov 22, 2017 2:31 PM
# author: mikahama
#SBATCH -J FOLDEP
#SBATCH --constraint="snb|hsw"
#SBATCH -p serial
#SBATCH --mem-per-cpu=4096
#SBATCH -n 4
#SBATCH -t 70:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mika.hamalainen@helsinki.fi

#module use -a /proj/nlpl/software/modulefiles/
module load python-env/2.7.13
module load java/oracle/1.8


cd /homeappl/home/mikahama/disambiguation-evaluation

source venv/bin/activate
python -E test_sctipt.py

used_slurm_resources.bash