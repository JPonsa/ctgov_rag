#!/bin/bash -l
#$ -N ctGov_eval_RAGAS
# Max run time in H:M:S
#$ -l h_rt=3:0:0
# Memory
#$ -l mem=80G
#$ -l gpu=1

# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module load openssl/1.1.1t python/3.11.3
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2
module load ruse/2.0

set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
MONGODB_USER=${MONGODB_USER//$'\r'}
MONGODB_PWD=${MONGODB_PWD//$'\r'}
LS_KEY=${LANGCHAIN_API_KEY//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}

pip install poetry
poetry install

# Track memory usage
ruse --stdout --time=150 -s \
poetry run python ./src/evaluation/RAGAS.py ./data/RAGA_testset.mistral_7b.csv \
    -user $MONGODB_USER -pwd $MONGODB_PWD -db ctGov -c trialgpt \
    -n 1 -size 2000 \
    -hf $HF_TOKEN \
    --generator mistralai/Mistral-7B-Instruct-v0.2 \
    --critic mistralai/Mistral-7B-Instruct-v0.2 \
    --embeddings sentence-transformers/all-mpnet-base-v2 \
    -test_size 5 -s 0.4 -r 0.4 -mc 0.2

#  --embeddings all-MiniLM-L6-v2 \