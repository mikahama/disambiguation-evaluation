import os
import glob
from subprocess import call

files = glob.glob("slurm/*.sh")
for file in files:
	call(["sbatch", file])
	print(file)

