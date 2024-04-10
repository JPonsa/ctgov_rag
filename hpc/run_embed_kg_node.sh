#!/bin/bash -l
#$ -N ctGov_KG_Embedding
# Max run time in H:M:S
#$ -l h_rt=10:0:0
# Memory
#$ -l mem=1G
#$ -pe smp 1
#$ -l gpu=2

# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2
module load openssl/1.1.1t python/3.11.3
module load ruse/2.0

pip install poetry
# poetry config virtualenvs.in-project true
# poetry install
ruse --stdout --time=900 -s  poetry run python embed_kg_nodes.py ./data/raw/knowledge_graph/ ./data/preprocessed/knowledge_graph/ 