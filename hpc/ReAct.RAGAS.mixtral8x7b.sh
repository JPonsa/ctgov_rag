#!/bin/bash -l
#$ -N ReAct_RAGAS_mixtral8x7b
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

MODEL=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
MODEL_NAME=mixtral8x7b
PORT=8073

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--quantization gptq \
--max-model-len 5000 \
--gpu-memory-utilization 0.80 &

echo I am going to sleep
sleep 5m # Go to sleep so I vLLM server has time to start.
echo I am awake

for mode in all llm_only sql_only cypher_only kg_only analytical_only; do
    
    echo $MODEL_NAME-$mode
    
    ruse --stdout --time=600 -s \
    poetry run python ./src/rag/ReAct.py -vllm $MODEL -port $PORT \
    -i ./data/preprocessed/RAGA_testset.llama3_8.manually_reviewed.tsv \
    -o ./results/ReAct/RAGAS.questioner.ReAct.$MODEL_NAME.$mode.tsv \
    -m $mode

done

echo ReAct $MODEL_NAME competed!
