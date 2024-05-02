#!/bin/bash -l
#$ -N neo4j_connection
# Max run time in H:M:S
#$ -l h_rt=0:20:0
# Memory
#$ -l mem=500M
#$ -pe smp 1  # see not askin for smp works with gpus


# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module load openssl/1.1.1t python/3.11.3
module load ruse/2.0


pip install poetry
poetry run python test_ne4j_connection.py