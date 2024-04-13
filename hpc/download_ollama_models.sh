#!/bin/bash -l
#$ -N Ollama_downloads
# Max run time in H:M:S
#$ -l h_rt=0:10:0
#$ -l mem=1G
#$ -pe smp 1
# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/


module load apptainer
export APPTAINER_TMPDIR=$XDG_RUNTIME_DIR/$USER_apptainerbuild
export APPTAINER_CACHEDIR=$HOME/Scratch/.apptainer
export OLLAMA_MODELS=$HOME/Scratch/ollama_models

apptainer instance start --net --network-args "portmap=11434:34/tcp" --fakeroot ollama.sif ollama_hpc
apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run mixtral:8x7b
apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run llama2:13b
apptainer exec --bind /tmp:$TMPDIR instance://ollama_hpc ollama run mistral