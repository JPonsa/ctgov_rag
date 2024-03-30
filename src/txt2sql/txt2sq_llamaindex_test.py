# Run txt2sql evaluation based on a give set of predefined questions
#
# Inputs:
# - /src/txt2sql/sql_queries_template.yaml
# Outputs:
# -/
# Usage: python ./src/txt2sql/txt2sql_llamaindex_test.py
#####################################################################

import os

import pandas as pd
import yaml
from dotenv import load_dotenv
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from requests.exceptions import ReadTimeout, Timeout
from sqlalchemy import create_engine
from tqdm import tqdm


def run_llamaindex_eval(query_engine, verbose: bool = True) -> pd.DataFrame:
    "Run txt-2-SQL evalution over a Llama-index SQL query engine"

    sql_eval_cols = [
        "question",
        "gold_std_query",
        "gold_std_answer",
        "llamaIndex_query",
        "llamaIndex_answer",
    ]
    sql_eval_rows = list(sql_queries_template.keys())
    sql_eval = pd.DataFrame([], index=sql_eval_rows, columns=sql_eval_cols)

    for q, d in tqdm(sql_queries_template.items(), desc="Evaluating llama index"):
        question = d["question"].format(
            nctId=nctId, condition=condition, intervention=intervention
        )
        sql_query = d["SQL"].format(
            nctId=nctId, condition=condition, intervention=intervention
        )

        if verbose:
            print(f"{q} : {question}")

        sql_eval.at[q, "question"] = question
        sql_eval.at[q, "gold_std_query"] = sql_query

        # Get gold standard answer
        try:
            answer = sql_db.run_sql(sql_query)[0]
        except:
            answer = "No answer"

        sql_eval.at[q, "gold_std_answer"] = answer

        # Get LlamaIndex SQL query and answer
        try:
            response = query_engine.query(question)
            sql_eval.at[q, "llamaIndex_query"] = response.metadata["sql_query"]
            sql_eval.at[q, "llamaIndex_answer"] = response.response
        except (ReadTimeout, Timeout, TimeoutError):
            if verbose:
                print("Time out!")
            sql_eval.at[q, "llamaIndex_query"] = "ReadTimeout"
            sql_eval.at[q, "llamaIndex_answer"] = "ReadTimeout"
        except Exception as e:
            if verbose:
                print(e)
            sql_eval.at[q, "llamaIndex_query"] = e
            sql_eval.at[q, "llamaIndex_answer"] = e

    return sql_eval


if __name__ == "__main__":

    # Experiment settings:
    condition = "Asthma"
    intervention = "Xhance"
    nctId = "NCT01164592"

    # Load SQL evaluation template
    with open("./src/txt2sql/sql_queries_template.yaml", "r") as file:
        sql_queries_template = yaml.safe_load(file)

    # Set Llama-index to run with Mixtral
    ## adapt the llama-index prompts to follow the
    ## Mistral format
    def completion_to_prompt(completion: str) -> str:
        return "[INST] {completion} [/INST] "

    def messages_to_prompt(messages) -> str:
        messages_str = [str(x) + "\n" for x in messages]
        return "[INST] {completion} [/INST] "

    lm = Ollama(
        model="mixtral",
        temperature=0.0,
        request_timeout=100,
        completion_to_prompt=completion_to_prompt,
        messages_to_prompt=messages_to_prompt,
    )
    Settings.llm = lm
    Settings.embed_model = "local"  # Local embedding instead of OpenAI emb.

    # Set a connection to the AACT SQL DB
    ## AACT credentials
    database = "aact"
    host = "aact-db.ctti-clinicaltrials.org"
    AACT_USER = os.getenv("AACT_USER")
    AACT_PWD = os.getenv("AACT_PWD")
    port = 5432
    db_uri = f"postgresql+psycopg2://{AACT_USER}:{AACT_PWD}@{host}:{port}/{database}"

    ## Subset of AACT relevant tables
    tables = [
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

    db_engine = create_engine(db_uri)
    sql_db = SQLDatabase(db_engine, include_tables=tables)
    # Standard query engine
    std_query_engine = NLSQLTableQueryEngine(sql_database=sql_db)

    # Advance query engine
    table_node_mapping = SQLTableNodeMapping(sql_db)
    table_schema_objs = [(SQLTableSchema(table_name=t)) for t in tables]

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
    sql_eval = run_llamaindex_eval(std_query_engine)
    sql_eval.to_csv(
        f"./results/txt2sql/llamaindex.{lm.dict()['model']}.TableQuery.eval.tsv",
        sep="\t",
    )

    ## For advance query engine.
    sql_eval = run_llamaindex_eval(adv_query_engine)
    sql_eval.to_csv(
        f"./results/txt2sql/llamaindex.{lm.dict()['model']}.TableRetriever.eval.tsv",
        sep="\t",
    )
