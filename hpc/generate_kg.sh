#!/bin/bash -l
#$ -N generate_kg
# Max run time in H:M:S
#$ -l h_rt=4:00:0
# Memory
#$ -l mem=500M
#$ -pe smp 1  # see not askin for smp works with gpus


# workig directory. Use #S -cwd to use current working dir
#$ -wd /home/rmhijpo/Scratch/ctgov_rag/

set -o allexport
source .env set

# remove special character (likely added as the file was created on a Windows)
MONGODB_USER=${MONGODB_USER//$'\r'}
MONGODB_PWD=${MONGODB_PWD//$'\r'}

module load openssl/1.1.1t python/3.11.3
pip install poetry

# poetry run python ./src/data/ctgov_to_mongodb.py --user $MONGODB_USER --pwd $MONGODB_PWD --database ctGov --collection trialgpt --studies ./data/trialgtp.studies_list.csv
poetry run python ./src/data/trim_clinical_trial.py --user $MONGODB_USER --pwd $MONGODB_PWD --database ctGov --collections "trialgpt" --metadata ./docs/ctGov.metadata.xlsx --overwrite
poetry run python ./src/knowledge_graph/create_knowledge_graph.py
poetry run python ./src/knowledge_graph/trim_node_label.py