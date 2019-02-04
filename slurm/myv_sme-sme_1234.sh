#!/bin/bash -l
# created: Nov 22, 2017 2:31 PM
# author: mikahama
#SBATCH -J "myv sme-sme-1234"
#SBATCH --constraint="snb|hsw"
#SBATCH -p serial
#SBATCH --mem-per-cpu=4096
#SBATCH -n 8
#SBATCH -t 70:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mika.hamalainen@helsinki.fi

#module use -a /proj/nlpl/software/modulefiles/
module load python-env/2.7.13
module load java/oracle/1.8
module unload hfst


cd /homeappl/home/mikahama/disambiguation-evaluation

source venv/bin/activate
for i in {1..10}
do
   python test_rnn_script.py --train_lang myv sme --test_lang sme --seed 1234 --model_filepath "models/myv sme-sme-1234_$i" --result_filepath "results/myv sme-sme-1234_$i"
done

used_slurm_resources.bash