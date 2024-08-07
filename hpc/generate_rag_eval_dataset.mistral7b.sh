#!/bin/bash -l
#$ -N RAGAS_mistral7b
# Max run time in H:M:S
#$ -l h_rt=8:0:0
# Memory
#$ -l mem=50G
#$ -l gpu=1
#$ -ac allow=EFL

# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module purge
module load openssl/1.1.1t python/3.11.3
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load cuda/12.2.2/gnu-10.2.0 
module load ruse/2.0

set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
MONGODB_USER=${MONGODB_USER//$'\r'}
MONGODB_PWD=${MONGODB_PWD//$'\r'}
LS_KEY=${LANGCHAIN_API_KEY//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}


MODEL=mistralai/Mistral-7B-Instruct-v0.2
PORT=8031

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.80 &

echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
# sleep 40m # Time to download
echo I am awake

ruse --stdout --time=150 -s \
poetry run python ./src/evaluation/RAGAS.py ./data/RAGA_testset.mistral7b.csv \
    -user $MONGODB_USER -pwd $MONGODB_PWD -db ctGov -c trialgpt \
    -n 25 -size 2000 \
    -hf $HF_TOKEN \
    -ports $PORT $PORT \
    --generator $MODEL \
    --critic $MODEL \
    --embeddings all-MiniLM-L6-v2 \
    -test_size 25 -s 0.4 -r 0.4 -mc 0.2
