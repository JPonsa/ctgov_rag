# Run txt2sql evaluation based on a give set of predefined questions
#
# Inputs:
# - /src/txt2sql/sql_queries_template.yaml
# Outputs:
# -/
# Usage: python ./src/txt2sql/txt2sql_llamaindex_test.py
#####################################################################

import argparse

import pandas as pd
import yaml
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from requests.exceptions import ReadTimeout, Timeout
from sqlalchemy import create_engine
from tqdm import tqdm

TABLES = [
    "browse_interventions",
    "sponsors",
    "detailed_descriptions",
    "facilities",
    "studies",
    "outcomes",
    "browse_conditions",
    "keywords",
    "eligibilities",
    "reported_events",
    "brief_summaries",
    "designs",
    "countries",
]

DATABASE = "aact"
HOST = "aact-db.ctti-clinicaltrials.org"
PORT = 5432


def generate_function(stop):
    def completion_to_prompt(completion: str) -> str:
        return f"{stop[0]} {completion} {stop[1]} "

    def messages_to_prompt(messages: list) -> str:
        messages_str = "\n".join(str(x) for x in messages)
        return f"{stop[0]} {messages_str} {stop[1]} "

    return completion_to_prompt, messages_to_prompt


def run_llamaindex_eval(
    query_engine: NLSQLTableQueryEngine | SQLTableRetrieverQueryEngine,
    sql_db: SQLDatabase,
    sql_queries_template: list[dict],
    triplets: list[tuple],
    verbose: bool = False,
) -> pd.DataFrame:
    "Run txt-2-SQL evaluation over a Llama-index SQL query engine"

    sql_eval_cols = [
        "question",
        "gold_std_query",
        "gold_std_answer",
        "llamaIndex_query",
        "llamaIndex_answer",
    ]
    sql_eval_rows = list(sql_queries_template.keys())
    sql_eval = pd.DataFrame([], columns=sql_eval_cols)

    for nctId, condition, intervention in triplets:
        tmp = pd.DataFrame([], index=sql_eval_rows, columns=sql_eval_cols)
        for q, d in tqdm(sql_queries_template.items(), desc="Evaluating llama index"):
            question = d["question"].format(
                nctId=nctId, condition=condition, intervention=intervention
            )
            sql_query = d["SQL"].format(
                nctId=nctId, condition=condition, intervention=intervention
            )

            if verbose:
                print(f"{q} : {question}")

            tmp.at[q, "question"] = question
            tmp.at[q, "gold_std_query"] = sql_query

            # Get gold standard answer
            try:
                answer = sql_db.run_sql(sql_query)[0]
            except:
                answer = "No answer"

            tmp.at[q, "gold_std_answer"] = answer

            # Get LlamaIndex SQL query and answer
            try:
                response = query_engine.query(question)
                tmp.at[q, "llamaIndex_query"] = response.metadata["sql_query"]
                tmp.at[q, "llamaIndex_answer"] = response.response
            except (ReadTimeout, Timeout, TimeoutError):
                if verbose:
                    print("Time out!")
                tmp.at[q, "llamaIndex_query"] = "ReadTimeout"
                tmp.at[q, "llamaIndex_answer"] = "ReadTimeout"
            except Exception as e:
                if verbose:
                    print(e)
                tmp.at[q, "llamaIndex_query"] = e
                tmp.at[q, "llamaIndex_answer"] = e

        sql_eval = pd.concat([sql_eval, tmp], ignore_index=True)

    return sql_eval


def main(args):

    # Load SQL evaluation template
    with open(args.sql_query_template, "r") as f:
        sql_queries_template = yaml.safe_load(f)

    with open(args.triplets, "r") as f:
        header = f.readline()
        triplets = f.readlines()

    triplets = [t.rstrip("\n").split("\t") for t in triplets]

    # Set LLM
    completion_to_prompt, messages_to_prompt = generate_function(args.stop)

    lm = Ollama(
        model=args.llm,
        temperature=0.0,
        request_timeout=100,
        completion_to_prompt=completion_to_prompt,
        messages_to_prompt=messages_to_prompt,
    )
    Settings.llm = lm
    Settings.embed_model = "local"

    # Set SQL DB connection
    db_uri = f"postgresql+psycopg2://{args.user}:{args.pwd}@{HOST}:{PORT}/{DATABASE}"
    db_engine = create_engine(db_uri)
    sql_db = SQLDatabase(db_engine, include_tables=TABLES)

    # Standard query engine
    std_query_engine = NLSQLTableQueryEngine(sql_database=sql_db)

    # Advance query engine
    table_node_mapping = SQLTableNodeMapping(sql_db)
    table_schema_objs = [(SQLTableSchema(table_name=t)) for t in TABLES]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    adv_query_engine = SQLTableRetrieverQueryEngine(
        sql_db, obj_index.as_retriever(similarity_top_k=3)
    )

    # Run evaluation
    ## For standard query engine
    sql_eval = run_llamaindex_eval(
        std_query_engine, sql_db, sql_queries_template, triplets
    )
    sql_eval.to_csv(
        f"{args.output_dir}llamaindex.{args.llm}.TableQuery.eval.tsv",
        sep="\t",
    )

    ## For advance query engine.
    sql_eval = run_llamaindex_eval(
        adv_query_engine, sql_db, sql_queries_template, triplets
    )
    sql_eval.to_csv(
        f"{args.output_dir}llamaindex.{args.llm}.TableRetriever.eval.tsv",
        sep="\t",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test llamaindex txt2sql")

    # Add arguments
    parser.add_argument("-user", type=str)
    parser.add_argument("-pwd", type=str)
    parser.add_argument("-sql_query_template", type=str)
    parser.add_argument("-triplets", type=str)
    parser.add_argument("--output_dir", type=str, default="./results/txt2sql/")
    parser.add_argument(
        "--llm", type=str, default="mistral", help="Ollama Large Language Model"
    )
    parser.add_argument("--stop", type=list, nargs="+", default=["INST", "/INST"])

    args = parser.parse_args()
    main(args)
