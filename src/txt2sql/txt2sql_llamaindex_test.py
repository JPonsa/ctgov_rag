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

AACT_TABLES = [
    "browse_interventions",
    "interventions",
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

# TODO: Review how this needs to be implemented property
def generate_prompt_adapter_func(stop: list = ["<s>[INST]", "[/INST] </s>"]):
    "Takes a list of stop tokens, produces functions to the llamaindex prompts"

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
    """Takes a SQL query engine and tests a the answering of questions in natural languange

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
        "gold_std_output",
        "llm_query",
        "llm_output",
        "llm_answer",
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

            tmp.at[q, "gold_std_output"] = answer.replace("\n", "|")

            # Get LlamaIndex SQL query and answer
            try:
                response = query_engine.query(question)
                llamaIndex_query = response.metadata["sql_query"]
                tmp.at[q, "llm_query"] = llamaIndex_query.replace("\n", " ")
                tmp.at[q, "llm_answer"] = response.response.replace("\n", "|")
            except (ReadTimeout, Timeout, TimeoutError):
                if verbose:
                    print("Time out!")
                tmp.at[q, "llm_query"] = "ReadTimeout"
                tmp.at[q, "llm_answer"] = "ReadTimeout"
            except Exception as e:
                if verbose:
                    print(f"Error - {e}")
                tmp.at[q, "llm_query"] = f"Error - {e}"
                tmp.at[q, "llm_answer"] = f"Error - {e}"
                
            try:
                tmp.at[q, "llm_output"] = sql_db.run_sql(response.metadata["sql_query"])[0].replace("\n", " ")
            except Exception as e:
                tmp.at[q, "llm_output"] = "No output"

        sql_eval = pd.concat([sql_eval, tmp], ignore_index=True)

    return sql_eval


def main(args, verbose: bool = False):
    
    file_tags = ["llamaindex", "w_completion_propmt"]

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
    
    if verbose:
        print(completion_to_prompt("completion_to_prompt test"))
    #     print(messages_to_prompt(["messages_to_prompt", "test"]))

    if args.hf:
        os.environ["HUGGING_FACE_TOKEN"] = args.hf
    
    if args.vllm:
        from llama_index.llms.openai_like import OpenAILike
        lm = OpenAILike(model=args.vllm, api_base=f"{args.host}:{args.port}/v1/", api_key="fake", temperature=0, max_tokens=1_000,
                        completion_to_prompt=completion_to_prompt)        
        file_tags.append(args.vllm.split("/")[-1])


    elif args.ollama:
        from llama_index.llms.ollama import Ollama

        lm = Ollama(
            model=args.ollama,
            temperature=0.0,
            request_timeout=100,
            # completion_to_prompt=completion_to_prompt,
            # messages_to_prompt=messages_to_prompt,
        )
        
        file_tags.append(args.ollama)
        
        
    Settings.llm = lm
    Settings.embed_model = "local"

    # Set SQL DB connection
    db_uri = f"postgresql+psycopg2://{args.user}:{args.pwd}@{HOST}:{PORT}/{DATABASE}"
    db_engine = create_engine(db_uri)
    sql_db = SQLDatabase(db_engine, include_tables=AACT_TABLES)

    # Standard query engine
    std_query_engine = NLSQLTableQueryEngine(sql_database=sql_db)

    # Advance query engine
    table_node_mapping = SQLTableNodeMapping(sql_db)
    table_schema_objs = [(SQLTableSchema(table_name=t)) for t in AACT_TABLES]

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
        f"{args.output_dir}{'.'.join(file_tags)}.TableQuery.eval.tsv",
        sep="\t",
    )

    ## For advance query engine.
    if verbose:
        print("Testing SQLTableRetrieverQueryEngine ...")
        
    sql_eval = run_llamaindex_eval(
        adv_query_engine, sql_db, sql_queries_templates, triplets, verbose
    )
    sql_eval.to_csv(
        f"{args.output_dir}{'.'.join(file_tags)}.TableRetriever.eval.tsv",
        sep="\t",
    )

    print("Testing LLamaindex txt2sql completed !")

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
        help="HuggingFace Token.",
    )
    
    parser.add_argument(
        "-vllm",
        default=argparse.SUPPRESS,
        help="Large Language Model name using HF nomenclature. E.g. 'mistralai/Mistral-7B-Instruct-v0.2'.",
    )
    
    parser.add_argument(
        "-host",
        type=str,
        default="http://0.0.0.0",
        help="LLM server host.",
    )

    parser.add_argument(
        "-port",
        type=int,
        default=8000,
        help="LLM server port.",
    )

    parser.add_argument(
        "-ollama",
        type=str,
        default="mistral",
        help="Large Language Model name using Ollama nomenclature. Default: 'mistral'.",
    )
    # TODO: Review, as it is not fully understood
    parser.add_argument(
        "-stop", type=str, nargs="+", default=["INST", "/INST"], help=""
    )

    parser.set_defaults(hf=None, vllm=None)

    args = parser.parse_args()
    main(args, verbose=True)
