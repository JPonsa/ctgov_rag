#!/bin/bash -l
#$ -N Eval_ReAct_mixtral8x7b
# Max run time in H:M:S
#$ -l h_rt=2:00:0
# Memory
#$ -l mem=64G
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

pip install poetry

MODEL_NAME=mixtral8x7b 


# for MODE in all llm_only sql_only cypher_only kg_only analytical_only; do
#     echo $MODEL_NAME-$mode
#     ruse --stdout --time=600 -s \
#     poetry run python ./src/evaluation/ctGov_questioner_eval.py \
#     -i ./results/ReAct/ctGov.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
#     -o ./results/eval/ctGov.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
#     -y answer -yhat ReAct_answer

#     echo Eval-ctGov.questioner-$MODEL_NAME-$MODE - completed
# done

# #for MODE in all llm_only sql_only cypher_only kg_only analytical_only; do
# for MODE in llm_only sql_only cypher_only analytical_only; do
#     echo $MODEL_NAME-$mode
#     ruse --stdout --time=600 -s \
#     poetry run python ./src/evaluation/ctGov_questioner_eval.py \
#     -i ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
#     -o ./results/eval/RAGAS.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
#     -y ground_truth -yhat ReAct_answer

#     echo Eval-RAGAS.questioner-$MODEL_NAME-$MODE - completed
# done

for MODE in llm_only cypher_only kg_only;
do
    echo Eval-trialgpt.questioner-$MODEL_NAME-all - start
    ruse --stdout --time=600 -s \
    poetry run python ./src/evaluation/trialgpt_questioner_eval.py \
    -i ./results/ReAct/trialgpt.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
    -o ./results/eval/trialgpt.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
    -y 2 -yhat ReAct_answer

    echo Eval-trialgpt.questioner-$MODEL_NAME-$MODE - completed

    echo Eval-trialgpt.questioner-$MODEL_NAME-all - start
    ruse --stdout --time=600 -s \
    poetry run python ./src/evaluation/trialgpt_questioner_eval.py \
    -i ./results/ReAct/trialgpt.questioner.ReAct_hint.$MODEL_NAME.$MODE.tsv \
    -o ./results/eval/trialgpt.questioner.ReAct_hint.$MODEL_NAME.$MODE.tsv \
    -y 2 -yhat ReAct_answer

    echo Eval-trialgpt.questioner-$MODEL_NAME-$MODE - completed
done