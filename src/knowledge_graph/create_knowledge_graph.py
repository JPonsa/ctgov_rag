import os
import shutil

from biocypher import BioCypher
from biocypher_adapter.ctgov_adapter import ctGovAdapter
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv("../../.env")

    MONGODB_USER = os.getenv("MONGODB_USER")
    MONGODB_PWD = os.getenv("MONGODB_PWD")

    biocypher_config_dir = "./src/knowledge_graph/biocypher_config/"
    output_dir = "./data/raw/knowledge_graph/"

    # ---- Create or clear output dir
    if os.path.exists(output_dir):  # Check if directory exists
        try:
            shutil.rmtree(output_dir)  # Delete directory
            print(f"Directory '{output_dir}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory '{output_dir}': {e}")

    try:
        os.makedirs(output_dir)
    except OSError as e:
        print(f"Failed to create directory {output_dir}. Reason: {e}")

    # ---- Create Knowledge Graph using Biocypher
    bc = BioCypher(
        biocypher_config_path=biocypher_config_dir + "biocypher_config.yaml",
        schema_config_path=biocypher_config_dir + "schema_config.yaml",
        output_directory=output_dir,
    )

    # ---- Use custom Biocypher adaptor for clinical trials stored in MongoDB
    adapter = ctGovAdapter(
        mongodb_user=MONGODB_USER,
        mongodb_pwd=MONGODB_PWD,
        mongodb_db="ctGov",
        mongodb_collection="trialgpt",
    )

    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())

    bc.write_import_call()

    # ---- Convert neoj4 admin command to windows compatible and neoj dbms 5.x
    with open(output_dir + "neo4j-admin-import-call.sh", "r") as f:
        neo4j_command = f.readline()

    neo4j_command = (
        neo4j_command.replace("bin/neo4j-admin", "bin\\neo4j-admin")
        .replace("--quote='\"'", "")
        .replace("import --database", "database import full")
        .replace("--force=true", "--overwrite-destination")
    )

    with open(output_dir + "neo4j-admin-import-call-windows.sh", "w") as f:
        f.write(neo4j_command)

# bc.summary() #BUG: Somehow takes too long - Raise ticket
# TODO: the
