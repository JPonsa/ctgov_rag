import os
import subprocess

from dotenv import load_dotenv

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PWD = os.getenv("MONGODB_PWD")


# Get asthma studies
# subprocess.run(
#     [
#         ".\.venv\Scripts\python.exe",
#         "./src/data/ctgov_to_mongodb.py",
#         "--user",
#         MONGODB_USER,
#         "--pwd",
#         MONGODB_PWD,
#         "--database",
#         "ctGov",
#         "--collection",
#         "asthma",
#         "--studies",
#         "./data/aact.browse_conditions.asthma.csv",
#     ]
# )

# Get heart_failure studies
# subprocess.run(
#     [
#         ".\.venv\Scripts\python.exe",
#         "./src/data/ctgov_to_mongodb.py",
#         "--user",
#         MONGODB_USER,
#         "--pwd",
#         MONGODB_PWD,
#         "--database",
#         "ctGov",
#         "--collection",
#         "heart_failure",
#         "--studies",
#         "./data/aact.browse_conditions.heart_failure.csv",
#     ]
# )


# Get trialgpt studies
subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/data/ctgov_to_mongodb.py",
        "--user",
        MONGODB_USER,
        "--pwd",
        MONGODB_PWD,
        "--database",
        "ctGov",
        "--collection",
        "trialgpt",
        "--studies",
        "./data/trialgtp.studies_list.csv",
    ]
)


# Combine previous collections and remove unwanted fields
subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/data/trim_clinical_trial.py",
        "--user",
        MONGODB_USER,
        "--pwd",
        MONGODB_PWD,
        "--database",
        "ctGov",
        "--collections",
        "trialgpt",
        "--metadata",
        "./docs/ctGov.metadata.xlsx",
        "--overwrite",
    ]
)

subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/knowledge_graph/create_knowledge_graph.py",
    ]
)


subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/knowledge_graph/trim_node_label.py",
    ]
)
