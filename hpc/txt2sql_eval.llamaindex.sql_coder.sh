#!/bin/bash -l
#$ -N sqlcoder_txt2SQL_eval
# Max run time in H:M:S
#$ -l h_rt=1:20:0
# Memory
#$ -l mem=10G
#$ -l gpu=1


# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module purge
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/3.11
module load ruse/2.0

set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
AACT_USER=${AACT_USER//$'\r'}
AACT_PWD=${AACT_PWD//$'\r'}
HF_TOKEN=${HF_TOKEN//$'\r'}


pip install poetry
ruse --stdout --time=150 -s \
poetry run python ./src/txt2sql/txt2sql_llamaindex_test.py -user $AACT_USER -pwd $AACT_PWD \
-sql_query_template ./src/txt2sql/sql_queries_template.yaml \
-triplets  ./src/txt2sql/txt2_sql_eval_triplets.tsv \
-output_dir ./results/txt2sql/ \
-hf $HF_TOKEN \
-vllm defog/sqlcoder-7b-2 \
-stop '' ''
