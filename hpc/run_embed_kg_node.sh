#!/bin/bash -l
#$ -N ctGov_KG_Embedding
# Max run time in H:M:S
#$ -l h_rt=2:0:0
#$ -pe smp 1
# Memory
#$ -l mem=32G
# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/
module load openssl/1.1.1t python/3.11.3
pip install poetry
# poetry config virtualenvs.in-project true
# poetry install
poetry run python embed_kg_nodes.py ./data/raw/knowledge_graph/ ./data/preprocessed/knowledge_graph/ 