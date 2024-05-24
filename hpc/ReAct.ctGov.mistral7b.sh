#!/bin/bash -l
#$ -N ReAct_ctGov_mistral
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=48G
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
AACT_USER=${AACT_USER//$'\r'}
AACT_PWD=${AACT_PWD//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}

MODEL=mistralai/Mistral-7B-Instruct-v0.2
MODEL_NAME=mistral7b
PORT=8042

export VLLM_TRACE_FUNCTION=1

pip install poetry
echo #---- Enviromental config
poetry run python collect_env.py
echo #------------------------
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.90 &
# --max-model-len 7500 \

echo $MODEL_NAME-llm_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.llm_only.tsv \
-m llm_only

echo $MODEL_NAME-sql_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.sql_only.tsv \
-m sql_only

echo $MODEL_NAME-cypher_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.cypher_only.tsv \
-m cypher_only

echo $MODEL_NAME-kg_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.kg_only.tsv \
-m kg_only

echo $MODEL_NAME-all
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.all.tsv \
-m all

echo ReAct $MODEL_NAME competed!