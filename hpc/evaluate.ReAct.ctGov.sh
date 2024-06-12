#!/bin/bash -l
#$ -N Eval_ReAct_ctGov
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=32G
#$ -l gpu=2
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

MODEL=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
MODEL_NAME=mixtral8x7b

pip install poetry

echo Eval-ctGov.questioner-$MODEL_NAME-all - start
ruse --stdout --time=600 -s \
poetry run python ./src/evaluation/ctGov_questioner_eval.py \
-i ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.all.tsv \
-o ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.all.eval.tsv \
-y answer -yhat ReAct_answer

echo Eval-ctGov.questioner-$MODEL_NAME-all - completed