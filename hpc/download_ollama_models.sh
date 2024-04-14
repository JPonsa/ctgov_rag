#!/bin/bash -l
#$ -N Ollama_downloads
# Max run time in H:M:S
#$ -l h_rt=0:10:0
#$ -l mem=1G
#$ -l gpu=1
# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/7.5.18/gnu-4.9.2
module load apptainer

export APPTAINER_TMPDIR=$XDG_RUNTIME_DIR/$USER_apptainerbuild
export APPTAINER_CACHEDIR=$HOME/Scratch/.apptainer
export OLLAMA_MODELS=$HOME/Scratch/ollama_models

apptainer exec --bind /tmp:$TMPDIR ollama.sif ollama run mixtral:8x7b
# apptainer instance start --net --network-args "portmap=11434:34/tcp" ollama.sif ollama_hpc
# apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run mixtral:8x7b
# apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run llama2:13b
# apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run mistral