#!/bin/bash -l
#$ -N mistral_txt2SQL_eval
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=30G
#$ -l gpu=1


# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module load openssl/1.1.1t python/3.11.3
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2
# module load ruse/2.0
# module load apptainer

set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
AACT_USER=${AACT_USER//$'\r'}
AACT_PWD=${AACT_PWD//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}


pip install poetry

# Arize tracing
# ssh -L 6006:myriad.rc.ucl.ac.uk:6006 rmhijpo@ssh-gateway.ucl.ac.uk -f
# wget https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py
# echo as is
# python collect_env.py
# echo from poetry
# poetry run python collect_env.py
echo before vllm
poetry run python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --dtype half &
echo after vllm
# apptainer instance start --nv ollama.sif ollama
# ruse --stdout --time=150 -s \
poetry run python ./src/txt2sql/txt2sql_dspy_test.py -user $AACT_USER -pwd $AACT_PWD \
-sql_query_template ./src/txt2sql/sql_queries_template.yaml \
-triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
-output_dir ./results/txt2sql/ \
-hf $HF_TOKEN \
-llm mistralai/Mistral-7B-Instruct-v0.2 \
-stop '[INST]' '[/INST]'
# -stop '[INST]' '[/INST]', '<<SYS>>', '<</SYS>>'