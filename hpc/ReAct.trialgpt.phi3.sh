#!/bin/bash -l
#$ -N ReAct_trialgpt_phi3
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


# MODEL=microsoft/Phi-3-mini-128k-instruct
# MODEL_NAME=phi3_m_128k
PORT=8065
MODEL=microsoft/Phi-3-mini-4k-instruct
MODEL_NAME=phi3_m_4k

pip install poetry

poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.90 &

echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake


for MODE in llm_only cypher_only kg_only; do

    echo $MODEL_NAME-$MODE

    ruse --stdout --time=600 -s \
    poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
    --context_max_tokens 1000 \
    -i ./data/trialgpt.questioner.tsv \
    -o ./results/ReAct/trialgpt.questioner.ReAct.$MODEL_NAME.$MODE.tsv \
    -m $MODE

    echo $MODEL_NAME-$MODE-hint

    ruse --stdout --time=600 -s \
    poetry run python ./src/rag/ReAct.trialgpt.py -vllm $MODEL -port $PORT \
    --context_max_tokens 1000 \
    -i ./data/trialgpt.questioner.tsv \
    -o ./results/ReAct/trialgpt.questioner.ReAct_hint.$MODEL_NAME.$MODE.tsv \
    -m $MODE \
    --hint

done

echo ReAct $MODEL_NAME competed!