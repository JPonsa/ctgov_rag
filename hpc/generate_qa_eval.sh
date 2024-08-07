#!/bin/bash -l
#$ -N QA_gen
# Max run time in H:M:S
#$ -l h_rt=0:30:0
# Memory
#$ -l mem=32G
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


MODEL=mistralai/Mistral-7B-Instruct-v0.2
PORT=8051

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model $MODEL --trust-remote-code --port $PORT --dtype half --enforce-eager \
--gpu-memory-utilization 0.95 &
#--quantization gptq \

echo I am going to sleep
sleep 1m # Go to sleep so I vLLM server has time to start.
echo I am awake
ruse --stdout --time=600 -s \
poetry run python ./src/evaluation/ctGov_QA_generator.py -vllm $MODEL -port $PORT -output_file ctGov.questioner.mistral7b.tsv
