#!/bin/bash
#PBS -N myjob
#PBS -q gpu_v100_q
#PBS -m abe
#PBS -M myemail@example.com
#PBS  -l select=1:ncpus=1
#PBS  -l select=mem=32 GB
#PBS  -l walltime=00:01:00
#PBS -o $HOME/mydir/output.log
#PBS -e $HOME/mydir/error.log
PBS_O_WORKDIR=$HOME/mydir
cd $PBS_O_WORKDIR

cd /path/to/working/directory

echo "Hello, world!" > output.txt
