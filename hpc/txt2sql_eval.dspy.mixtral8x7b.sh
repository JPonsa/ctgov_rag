#!/bin/bash -l
#$ -N dspy_mixtral8x7b_txt2SQL_eval
# Max run time in H:M:S
#$ -l h_rt=2:00:0
# Memory
#$ -l mem=100G
#$ -l gpu=2


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
AACT_USER=${AACT_USER//$'\r'}
AACT_PWD=${AACT_PWD//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}

MODEL=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
PORT=8001

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --port $PORT --dtype half --enforce-eager \
--quantization gptq \
--max-model-len 5000 \
--gpu-memory-utilization 0.80 &
echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake
ruse --stdout --time=600 -s \
poetry run python ./src/txt2sql/txt2sql_dspy_test.py -user $AACT_USER -pwd $AACT_PWD \
-sql_query_template ./src/txt2sql/sql_queries_template.yaml \
-triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
-output_dir ./results/txt2sql/ \
-hf $HF_TOKEN \
-vllm $MODEL \
-port $PORT
# -stop '[INST]' '[/INST]'
