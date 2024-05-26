#!/bin/bash -l
#$ -N ReAct_trialgpt_v2_llama3
# Max run time in H:M:S
#$ -l h_rt=12:00:0
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

MODEL=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
MODEL_NAME=mixtral8x7b
PORT=8053

pip install poetry
echo #---- Enviromental config
poetry run python collect_env.py
echo #------------------------
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--quantization gptq \
--max-model-len 5000 \
--gpu-memory-utilization 0.80 &

echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake

# echo $MODEL-all
# ruse --stdout --time=600 -s \
# poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
# -i ./data/trialgpt.questioner.tsv \
# -o ./results/ReAct/trialgpt.questioner.ReAct_v2.$MODEL_NAME.all.tsv \
# -m all

# echo $MODEL-sql_only
# ruse --stdout --time=600 -s \
# poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
# -i ./data/trialgpt.questioner.tsv \
# -o ./results/ReAct/trialgpt.questioner.ReAct_v2.$MODEL_NAME.sql_only.tsv \
# -m sql_only

echo $MODEL-kg_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
-i ./data/trialgpt.questioner.tsv \
-o ./results/ReAct/trialgpt.questioner.ReAct_v2.$MODEL_NAME.kg_only.tsv \
-m kg_only

echo $MODEL-cypher_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
-i ./data/trialgpt.questioner.tsv \
-o ./results/ReAct/trialgpt.questioner.ReAct_v2.$MODEL_NAME.cypher_only.tsv \
-m cypher_only

echo $MODEL-llm_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
-i ./data/trialgpt.questioner.tsv \
-o ./results/ReAct/trialgpt.questioner.ReAct_v2.$MODEL_NAME.llm_only.tsv \
-m llm_only

echo ReAct $MODEL_NAME competed!

# -user $AACT_USER -pwd $AACT_PWD \
# -sql_query_template ./src/txt2sql/sql_queries_template.yaml \
# -triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
# -output_dir ./results/txt2sql/ \
# -hf $HF_TOKEN \
# -vllm $MODEL \
# -port $PORT
# -stop '[INST]' '[/INST]'
