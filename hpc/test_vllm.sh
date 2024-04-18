#!/bin/bash -l
#$ -N test_vLLM
# Max run time in H:M:S
#$ -l h_rt=0:15:0
# Memory
#$ -l mem=30G
#$ -l gpu=1


# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

module load openssl/1.1.1t python/3.11.3
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2

pip install poetry
poetry run python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --dtype half &
echo I go to spleep
sleep 10m
echo I am awake

echo http://0.0.0.0:8000/v1/chat/completions
curl http://0.0.0.0:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mistralai/Mistral-7B-Instruct-v0.2",
"messages": [
{"role": "user", "content": "Who won the world series in 2020?"}
]
}'

echo http://localhost:8000/v1/chat/completions
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mistralai/Mistral-7B-Instruct-v0.2",
"messages": [
{"role": "user", "content": "What is the capital of France?"}
]
}'



echo myriad.rc.ucl.ac.uk:8000/v1/chat/completions 
curl myriad.rc.ucl.ac.uk:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mistralai/Mistral-7B-Instruct-v0.2",
"messages": [
{"role": "user", "content": "Why the grass is green?"}
]
}'


echo http://myriad.rc.ucl.ac.uk:8000/v1/chat/completions 
curl http://myriad.rc.ucl.ac.uk:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "mistralai/Mistral-7B-Instruct-v0.2",
"messages": [
{"role": "user", "content": "Why the sky is blue?"}
]
}'
