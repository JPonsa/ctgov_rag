#!/bin/bash -l
#$ -N ReAct_RAGAS_llama3
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

MODEL=meta-llama/Meta-Llama-3-8B-Instruct
MODEL_NAME=llama3_8b
PORT=8075

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.80 &

echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake

echo $MODEL_NAME-llm_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
-o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.llm_only.tsv \
-m llm_only

echo $MODEL_NAME-sql_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
-o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.sql_only.tsv \
-m sql_only

echo $MODEL_NAME-cypher_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
-o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.cypher_only.tsv \
-m cypher_only

echo $MODEL_NAME-kg_only
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
-o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.kg_only.tsv \
-m kg_only

echo $MODEL_NAME-all
ruse --stdout --time=600 -s \
poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
-i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
-o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.all.tsv \
-m all

echo ReAct $MODEL_NAME competed!
