#!/bin/bash -l
#$ -N Eval_ReAct_ctGov
# Max run time in H:M:S
#$ -l h_rt=0:20:0
# Memory
#$ -l mem=10G
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
module load tensorflow/2.0.0/gpu-py37

set -o allexport
source .env set

pip install poetry

echo E$MODEL_NAME-llm_only
ruse --stdout --time=600 -s \
poetry run python ./src/evaluation/ctGov_questioner_eval.py \
-i ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.all.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.all.eval.tsv \
-y answer -yhat ReAct_answer

echo E$MODEL_NAME-ll