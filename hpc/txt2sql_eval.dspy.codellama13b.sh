#!/bin/bash -l
#$ -N codellama_txt2SQL_eval
# Max run time in H:M:S
#$ -l h_rt=1:00:0
# Memory
#$ -l mem=32G
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
AACT_USER=${AACT_USER//$'\r'}
AACT_PWD=${AACT_PWD//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}
MODEL=codellama/CodeLlama-13b-hf


pip install poetry

# Arize tracing
ssh -L 6006:myriad.rc.ucl.ac.uk:6006 rmhijpo@ssh-gateway.ucl.ac.uk -f
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --port 8000 --dtype half --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 2500 &
echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start
sleep 10m # Give time do download the model. Only needed the 1st time
echo I am awake
# apptainer instance start --nv ollama.sif ollama
ruse --stdout --time=600 -s poetry run python ./src/txt2sql/txt2sql_dspy_test.py -user $AACT_USER -pwd $AACT_PWD \
-sql_query_template ./src/txt2sql/sql_queries_template.yaml \
-triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
-output_dir ./results/txt2sql/ \
-hf $HF_TOKEN \
-vllm $MODEL \
-stop '[INST]' '[/INST]'
