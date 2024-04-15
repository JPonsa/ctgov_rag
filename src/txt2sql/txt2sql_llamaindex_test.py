import argparse
import os

import pandas as pd
import yaml
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.query_engine import NLSQLTableQueryEngine
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


def generate_prompt_adapter_func(stop: list = ["[INST]", "[/INST]"]):
    "Given a list of stop tokens, generates functions to the llamaindex prompts"

    def completion_to_prompt(completion: str) -> str:
        return f"{stop[0]} {completion} {stop[1]} "

    def messages_to_prompt(messages: list) -> str:
        messages_str = "\n".join(str(x) for x in messages)
        return f"{stop[0]} {messages_str} {stop[1]} "

    return completion_to_prompt, messages_to_prompt


def run_llamaindex_eval(
    query_engine: NLSQLTableQueryEngine | SQLTableRetrieverQueryEngine,
    sql_db: SQLDatabase,
    sql_queries_templates: list[dict],
    triplets: list[list[str]],
    verbose: bool = False,
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    query_engine : NLSQLTableQueryEngine | SQLTableRetrieverQueryEngine
        Llama-index SQL query engine
    sql_db : SQLDatabase
        SQL DB connection
    sql_queries_templates : list[dict]
        list of SQL query templates. Each template is composed of a question and a SQL query.
    triplets : list[list[str]]
        nctId, condition, intervention
    verbose : bool, optional
        print progression messages, by default False

    Returns
    -------
    pd.DataFrame
        _description_
    """

    sql_eval_cols = [
        "question",
        "gold_std_query",
        "gold_std_answer",
        "llamaIndex_query",
        "llamaIndex_answer",
    ]
    sql_eval_rows = list(sql_queries_templates.keys())
    sql_eval = pd.DataFrame([], columns=sql_eval_cols)

    for nctId, condition, intervention in triplets:
        if verbose:
            print(
                f"Triplet: nctId: {nctId} | condition: {condition} | intervention: {intervention}"
            )
        tmp = pd.DataFrame([], index=sql_eval_rows, columns=sql_eval_cols)
        for q, d in tqdm(sql_queries_templates.items(), desc="Evaluating llama index"):
            question = d["question"].format(
                nctId=nctId, condition=condition, intervention=intervention
            )
            sql_query = d["SQL"].format(
                nctId=nctId, condition=condition, intervention=intervention
            )

            if verbose:
                print(f"{q} : {question}")

            tmp.at[q, "question"] = question.replace("\n", "|")
            tmp.at[q, "gold_std_query"] = sql_query.replace("\n", " ")

            # Get gold standard answer
            try:
                answer = sql_db.run_sql(sql_query)[0]
            except:
                answer = "No answer"

            tmp.at[q, "gold_std_answer"] = answer.replace("\n", "|")

            # Get LlamaIndex SQL query and answer
            try:
                response = query_engine.query(question)
                llamaIndex_query = response.metadata["sql_query"]
                tmp.at[q, "llamaIndex_query"] = llamaIndex_query.replace("\n", " ")
                tmp.at[q, "llamaIndex_answer"] = response.response.replace("\n", "|")
            except (ReadTimeout, Timeout, TimeoutError):
                if verbose:
                    print("Time out!")
                tmp.at[q, "llamaIndex_query"] = "ReadTimeout"
                tmp.at[q, "llamaIndex_answer"] = "ReadTimeout"
            except Exception as e:
                if verbose:
                    print(f"Error - {e}")
                tmp.at[q, "llamaIndex_query"] = f"Error - {e}"
                tmp.at[q, "llamaIndex_answer"] = f"Error - {e}"

        sql_eval = pd.concat([sql_eval, tmp], ignore_index=True)

    return sql_eval


def main(args, verbose: bool = False):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load SQL evaluation template
    with open(args.sql_query_template, "r") as f:
        sql_queries_templates = yaml.safe_load(f)

    with open(args.triplets, "r") as f:
        header = f.readline()
        triplets = f.readlines()

    triplets = [t.rstrip("\n").split("\t") for t in triplets]

    # Set LLM
    completion_to_prompt, messages_to_prompt = generate_prompt_adapter_func(args.stop)

    if args.hf:
        os.environ["HUGGING_FACE_TOKEN"] = args.hf
        from llama_index.llms.huggingface import HuggingFaceLLM

        lm = HuggingFaceLLM(
            model_name=args.llm,
            tokenizer_name=args.llm,
            generate_kwargs={"temperature": 0.0},
            device_map="auto",
            # BUG: bitsandbytes not finding CUDA lib in HPC. Fix and activate
            model_kwargs={"load_in_4bit": False},
            completion_to_prompt=completion_to_prompt,
            messages_to_prompt=messages_to_prompt,
        )

    else:
        from llama_index.llms.ollama import Ollama

        lm = Ollama(
            model=args.llm,
            temperature=0.0,
            request_timeout=100,
            completion_to_prompt=completion_to_prompt,
            messages_to_prompt=messages_to_prompt,
        )
    Settings.llm = lm
    Settings.embed_model = "local"

    if verbose:
        print(f"Testing Llama-index with LLM {args.llm}")

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
    if verbose:
        print("Testing NLSQLTableQueryEngine ...")
    sql_eval = run_llamaindex_eval(
        std_query_engine, sql_db, sql_queries_templates, triplets, verbose
    )
    sql_eval.to_csv(
        f"{args.output_dir}llamaindex.{args.llm.split('/')[-1]}.TableQuery.eval.tsv",
        sep="\t",
    )

    ## For advance query engine.
    if verbose:
        print("Testing SQLTableRetrieverQueryEngine ...")
    sql_eval = run_llamaindex_eval(
        adv_query_engine, sql_db, sql_queries_templates, triplets, verbose
    )
    sql_eval.to_csv(
        f"{args.output_dir}llamaindex.{args.llm.split('/')[-1]}.TableRetriever.eval.tsv",
        sep="\t",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test llamaindex txt2sql")

    # Add arguments
    parser.add_argument("-user", type=str, help="AACT user name.")
    parser.add_argument("-pwd", type=str, help="AACT password.")
    parser.add_argument(
        "-sql_query_template",
        type=str,
        help="yaml file containing query templates. Each template contains a user question and associated SQL query. templates assume the presence of {nctId}, {condition} and {intervention}.",
    )
    parser.add_argument(
        "-triplets",
        type=str,
        help="TSV file containing nctId, condition, intervention triplets.",
    )

    parser.add_argument(
        "-output_dir",
        type=str,
        default="./results/txt2sql/",
        help="path to directory where to store results.",
    )

    parser.add_argument(
        "-hf",
        default=argparse.SUPPRESS,
        help="HuggingFace Token. If not provided, assumes that Ollama.",
    )

    parser.add_argument(
        "-llm",
        type=str,
        default="mistral",
        help="Large Language Model. E.g for Ollama use 'mistral' for HF use 'mistralai/Mistral-7B-Instruct-v0.2'",
    )
    parser.add_argument(
        "-stop", type=str, nargs="+", default=["INST", "/INST"], help=""
    )

    parser.set_defaults(hf=None)

    args = parser.parse_args()
    main(args, verbose=False)
