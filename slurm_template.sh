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

module use -a /proj/nlpl/software/modulefiles/
module load nlpl-opennmt-py


cd /homeappl/home/mikahama/nmt

python preprocess.py -train_src FOLDERold.txt -train_tgt FOLDERnew.txt -valid_src FOLDERold_valid.txt -valid_tgt FOLDERnew_valid.txt -save_data FOLDERdata/vocabulary

python train.py -data FOLDERdata/vocabulary -save_model FOLDERmodel/nmt-model

lastmodel=$( ls -t FOLDERmodel/*.pt | head -1 )

python translate.py -model $lastmodel -src test.txt -output FOLDERpred.txt -replace_unk

used_slurm_resources.bash