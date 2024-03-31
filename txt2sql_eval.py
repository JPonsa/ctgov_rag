import os
import subprocess

from dotenv import load_dotenv

load_dotenv(".env")

subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/txt2sql/txt2sql_llamaindex_test.py",
        "-user",
        os.getenv("AACT_USER"),
        "-pwd",
        os.getenv("AACT_PWD"),
        "-sql_query_template",
        "./src/txt2sql/sql_queries_template.yaml",
        "-triplets",
        "./src/txt2sql/txt2_sql_eval_triplets.tsv",
        "--output_dir",
        "./results/txt2sql/",
        "--llm",
        "sqlcoder",
        "--stop",
        "['', '']",
    ]
)

subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/txt2sql/txt2sql_llamaindex_test.py",
        "-user",
        os.getenv("AACT_USER"),
        "-pwd",
        os.getenv("AACT_PWD"),
        "-sql_query_template",
        "./src/txt2sql/sql_queries_template.yaml",
        "-triplets",
        "./src/txt2sql/txt2_sql_eval_triplets.tsv",
        "--output_dir",
        "./results/txt2sql/",
        "--llm",
        "mistral",
        "--stop",
        "['INST', '/INST']",
    ]
)

subprocess.run(
    [
        ".\.venv\Scripts\python.exe",
        "./src/txt2sql/txt2sql_llamaindex_test.py",
        "-user",
        os.getenv("AACT_USER"),
        "-pwd",
        os.getenv("AACT_PWD"),
        "-sql_query_template",
        "./src/txt2sql/sql_queries_template.yaml",
        "-triplets",
        "./src/txt2sql/txt2_sql_eval_triplets.tsv",
        "--output_dir",
        "./results/txt2sql/",
        "--llm",
        "codellama",
        "--stop",
        "['INST', '/INST']",
    ]
)
