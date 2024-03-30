

module load openssl/1.1.1t python/3.11.3

apptainer pull ollama.sif docker://ollama/ollama:latest
apptainer run ollama.sif

git clone https://github.com/JPonsa/ctogov_rag.git
cd ctogov_rag.git
pip install poetry
poetry install
python ./src/txt2sql/txt2sql_llamaindex_test.py