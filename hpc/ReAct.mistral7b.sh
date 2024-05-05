#!/bin/bash -l
#$ -N ReAct_mistral
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=48G
#$ -l gpu=1


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
PORT=8042

export VLLM_TRACE_FUNCTION=1

pip install poetry
echo #---- Enviromental config
poetry run python collect_env.py
echo #------------------------
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.90 &
# --max-model-len 7500 \
echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.mixtral7x8b.all.tsv \
-m all

poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.mixtral7x8b.sql_only.tsv \
-m sql_only

poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.mixtral7x8b.kg_only.tsv \
-m kg_only

poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/ctGov.questioner.mistral7b.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.mixtral7x8b.cypher_only.tsv \
-m cypher_only

# -user $AACT_USER -pwd $AACT_PWD \
# -sql_query_template ./src/txt2sql/sql_queries_template.yaml \
# -triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
# -output_dir ./results/txt2sql/ \
# -hf $HF_TOKEN \
# -vllm $MODEL \
# -port $PORT
# -stop '[INST]' '[/INST]'
