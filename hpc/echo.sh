set -o allexport
source .env set
# +o allexport

# set -a
# source .env
# set +a



MONGODB_USER=${MONGODB_USER//$'\r'}
MONGODB_PWD=${MONGODB_PWD//$'\r'}

echo poetry run python embed_kg_nodes.py ./data/raw/knowledge_graph/ ./data/preprocessed/knowledge_graph/ -mongoDB -user $MONGODB_USER -pwd $MONGODB_PWD -db ctGov -c trialgpt