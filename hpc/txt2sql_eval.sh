#!/bin/bash -l
#$ -N ctGov_txt2SQL_eval
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=64G
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


pip install poetry
# poetry lock
poetry install

ruse --stdout --time=150 -s \
poetry run python ./src/txt2sql/txt2sql_llamaindex_test.py -user $AACT_USER -pwd $AACT_PWD \
-sql_query_template ./src/txt2sql/sql_queries_template.yaml \
-triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
-output_dir ./results/txt2sql/ \
-hf $HF_TOKEN \
-llm mistralai/Mistral-7B-Instruct-v0.2 \
-stop "['INST', '/INST']"

# ruse --stdout --time=900 -s \
# poetry run python ./src/txt2sql/txt2sql_llamaindex_test.py -user $AACT_USER -pwd $AACT_PWD \
# -sql_query_template ./src/txt2sql/sql_queries_template.yaml \
# -triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
# -output_dir ./results/txt2sql/ \
# -hf $HF_TOKEN \
# -llm defog/sqlcoder \
# -stop "['', '']"