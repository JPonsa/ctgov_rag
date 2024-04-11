#!/bin/bash -l
#$ -N ctGov_KG_Embedding
# Max run time in H:M:S
#$ -l h_rt=12:00:0
# Memory
#$ -l mem=2G
#$ -pe smp 2  # see not askin for smp works with gpus

# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module load openssl/1.1.1t python/3.11.3
module load ruse/2.0


set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
MONGODB_USER=${MONGODB_USER//$'\r'}
MONGODB_PWD=${MONGODB_PWD//$'\r'}


pip install poetry
# poetry config virtualenvs.in-project true
# poetry install
ruse --stdout --time=900 -s poetry run python embed_kg_nodes.py ./data/raw/knowledge_graph/ ./data/preprocessed/knowledge_graph/ -mongoDB -user $MONGODB_USER -pwd $MONGODB_PWD -db ctGov -c trialgpt