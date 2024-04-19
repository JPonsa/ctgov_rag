import argparse
import os
import sys

import dspy
import pandas as pd
import yaml
from tqdm import tqdm

####### Add src folder to the system path so it can call utils
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.sql_wrapper import SQLDatabase
from utils.utils import dspy_tracing

# AACT Connection parameters
DATABASE = "aact"
HOST = "aact-db.ctti-clinicaltrials.org"
PORT = 5432

# Selection of relevant tables in AACT
AACT_TABLES = [
    "browse_interventions",
    "sponsors",
    "detailed_descriptions",
    "facilities",
    # "studies",
    "outcomes",
    "browse_conditions",
    "keywords",
    "eligibilities",
    "reported_events",
    "brief_summaries",
    "designs",
    "countries",
]

STUDY_TABLE = (
    "Table 'studies' has columns: nct_id (VARCHAR),"
    # "nlm_download_date_description (VARCHAR), "
    # "study_first_submitted_date (DATE), "
    # "results_first_submitted_date (DATE), "
    # "disposition_first_submitted_date (DATE), "
    # "last_update_submitted_date (DATE), "
    # "study_first_submitted_qc_date (DATE), "
    # "study_first_posted_date (DATE), "
    # "study_first_posted_date_type (VARCHAR), "
    # "results_first_submitted_qc_date (DATE), "
    # "results_first_posted_date (DATE), "
    # "results_first_posted_date_type (VARCHAR), "
    # "disposition_first_submitted_qc_date (DATE), "
    # "disposition_first_posted_date (DATE), "
    # "disposition_first_posted_date_type (VARCHAR), "
    # "last_update_submitted_qc_date (DATE), "
    # "last_update_posted_date (DATE), "
    # "last_update_posted_date_type (VARCHAR), "
    # "start_month_year (VARCHAR), "
    # "start_date_type (VARCHAR), "
    # "start_date (DATE), "
    # "verification_month_year (VARCHAR), "
    # "verification_date (DATE), "
    # "completion_month_year (VARCHAR), "
    # "completion_date_type (VARCHAR), "
    # "completion_date (DATE), "
    # "primary_completion_month_year (VARCHAR), "
    # "primary_completion_date_type (VARCHAR), "
    # "primary_completion_date (DATE), "
    # "target_duration (VARCHAR), "
    "study_type (VARCHAR), "
    # "acronym (VARCHAR), "
    "baseline_population (TEXT), "
    "brief_title (TEXT), "
    # "official_title (TEXT), "
    "overall_status (VARCHAR), "
    # "last_known_status (VARCHAR), "
    "phase (VARCHAR), "
    "enrollment (INTEGER), "
    # "enrollment_type (VARCHAR), "
    "source (VARCHAR), "
    "limitations_and_caveats (VARCHAR), "
    # "number_of_arms (INTEGER), "
    # "number_of_groups (INTEGER), "
    "why_stopped (VARCHAR), "
    # "has_expanded_access (BOOLEAN), "
    # "expanded_access_type_individual (BOOLEAN), "
    # "expanded_access_type_intermediate (BOOLEAN), "
    # "expanded_access_type_treatment (BOOLEAN), "
    # "has_dmc (BOOLEAN), "
    # "is_fda_regulated_drug (BOOLEAN), "
    # "is_fda_regulated_device (BOOLEAN), "
    # "is_unapproved_device (BOOLEAN), "
    # "is_ppsd (BOOLEAN), "
    # "is_us_export (BOOLEAN), "
    "biospec_retention (VARCHAR), "
    "biospec_description (TEXT), "
    # "ipd_time_frame (VARCHAR), "
    # "ipd_access_criteria (VARCHAR), "
    # "ipd_url (VARCHAR), "
    # "plan_to_share_ipd (VARCHAR), "
    # "plan_to_share_ipd_description (VARCHAR), "
    # "created_at (TIMESTAMP), "
    # "updated_at (TIMESTAMP), "
    # "source_class (VARCHAR), "
    # "delayed_posting (VARCHAR), "
    # "expanded_access_nctid (VARCHAR),"
    # "expanded_access_status_for_nctid (VARCHAR), "
    # "fdaaa801_violation (BOOLEAN), "
    # "baseline_type_units_analyzed (VARCHAR), "
    "and foreign keys: ['nct_id'] -> studies.['nct_id']"
)


# Common SQL mistakes
COMMON_MISTAKES = (
    "(1) Using NOT IN with NULL values. "
    "(2) Using UNION when UNION ALL should have been used. "
    "(3) Using BETWEEN for exclusive ranges. "
    "(4) Data type mismatch in predicates. "
    "(5) Not properly quoting identifiers. "
    "(6) Not using the correct number of arguments for functions. "
    "(7) Not casting to the correct data type. "
    "(8) Not using the proper columns for joins. "
    "(9) Not keeping the query as simples as possible. "
    "(10) Not using ILIKE instead of LIKE; ILIKE is always preferred over LIKE "
    "(11) Not making all WHERE statements case insensitive using LOWER."
)
    

class Text2Sql(dspy.Signature):
    """Given an input question and a SQL db schema, produce a syntactically correct PostgreSQL query to run.
    Never query for all the columns from a specific table, only ask for a few relevant columns given the question.
    Pay attention to use only the column names that you can see in the schema description.
    Be careful to not query for columns that do not exist. Pay attention to which column is in which table.
    Also, qualify column names with the table name when needed."""

    context: str = dspy.InputField(prefix="Schema:", desc="SQL db schema")
    question: str = dspy.InputField(prefix="Question:", desc="user question")
    sql_query: str = dspy.OutputField(
        prefix="SQL query:",
        desc="SQL query that answers the user question",
    )


class CheckSqlQuery(dspy.Signature):
    """Given a SQL query and a list of common mistakes, produce a revised SQL query."""

    context: str = dspy.InputField(
        prefix="Common mistakes:",
        desc="list of common mistakes",
    )
    sql_query: str = dspy.InputField(
        prefix="SQL query:",
        desc="original sql query",
    )
    revised_sql: str = dspy.OutputField(
        prefix="Revised SQL query:",
        desc="sql query correcting any common mistake",
    )


class CheckSqlSchema(dspy.Signature):
    """Given a SQL query and a SQL db schema, produce a revised SQL query."""

    context: str = dspy.InputField(prefix="SQL db schema:")
    sql_query: str = dspy.InputField(prefix="SQL query:", desc="original sql query")
    revised_sql: str = dspy.OutputField(
        prefix="Revised SQL query:",
        desc="revised sql query making sure all references align with the SQL schema.",
    )


class CheckSqlError(dspy.Signature):
    """Give a SQL query, a SQL schema and the error raised when running it, produce a revised SQL query."""

    db_schema: str = dspy.InputField(prefix="Schema:", desc="SQL DB schema")
    sql_query: str = dspy.InputField(prefix="SQL query:", desc="original SQL query")
    error: str = dspy.InputField(
        desc="Exception through when running  the original SQL query",
        prefix="Exception: ",
    )
    revised_sql = dspy.OutputField(
        prefix="Revised: ", desc="Revised SQL query that addresses the error"
    )


class QuestionSqlAnswer(dspy.Signature):
    """Given an input question and the result of a SQL query that could provide useful information regarding the question, produce an answer."""

    question: str = dspy.InputField(prefix="Question:", desc="user question")
    context: str = dspy.InputField(
        prefix="SQL output:",
        desc="SQL output that could provide useful information regarding the question",
    )
    answer: str = dspy.OutputField(prefix="Answer:", desc="answer to the user question")


class Txt2SqlAgent(dspy.Module):
    def __init__(
        self, sql_db: SQLDatabase, sql_schema: str, common_mistakes: str
    ) -> None:
        super().__init__()
        self.txt2sql = dspy.Predict(Text2Sql)
        self.review_query = dspy.Predict(CheckSqlQuery)
        self.review_schema = dspy.Predict(CheckSqlSchema)
        self.question_sql_answer = dspy.Predict(QuestionSqlAnswer)
        self.sql_db = sql_db
        self.sql_schema = sql_schema
        self.common_mistakes = common_mistakes

    def forward(self, question: str, n: int = 3, verbose:bool=False) -> str:
        response = {}
        attempts = 0
        sql_output = None

        response["txt2sql"] = self.txt2sql(
            context=self.sql_schema,
            question=question,
        )
        
        response["sql_query"] = response["txt2sql"].sql_query
        
        if verbose:
            print(f"Initial SQL query: {response['sql_query']}")

        while attempts < n and sql_output is None:
            try:
                sql_output = self.sql_db.run_sql(response["sql_query"])
            except Exception as e:
                
                if verbose:
                    print(str(e))

                response["review_query"] = self.review_query(
                    context=str(e),
                    sql_query = response["sql_query"],
                    )

                response["review_schema"] = self.review_schema(
                    context=self.sql_schema,
                    sql_query=response["review_query"].revised_sql,
                    )
                
                response["sql_query"] = response["review_schema"].revised_sql
                
                if verbose:
                    print(f"Revised SQL query attempt {attempts}: {response['sql_query']}")
                

                attempts += 1

        if not sql_output:
            sql_output = "Information not found."

        print(type(sql_output))

        response["final_answer"] = self.question_sql_answer(
            context=sql_output,
            question=question,
            )
        return response


def run_sql_eval(
    query_engine,
    sql_db: SQLDatabase,
    sql_queries_templates: list[dict],
    triplets: list[list[str]],
    verbose: bool = False,
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    query_engine : dspy model
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
        "llm_query",
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
        for q, d in tqdm(sql_queries_templates.items(), desc="Evaluating txt2sql dspy"):
                            
            question = d["question"].format(
                nctId=nctId,
                condition=condition,
                intervention=intervention,
            )
            
            sql_query = d["SQL"].format(
                nctId=nctId,
                condition=condition,
                intervention=intervention,
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
            
            # Get the answer from the LLM
            response = query_engine.forward(question, verbose=verbose)
            tmp.at[q, "llm_query"] = response["sql_query"].replace("\n", " ")
            tmp.at[q, "llm_answer"] = response["final_answer"].answer.replace("\n", "|")

        sql_eval = pd.concat([sql_eval, tmp], ignore_index=True)

    return sql_eval


def main(args, verbose: bool = False):
    
    file_tags = ["dspy"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load SQL evaluation template
    with open(args.sql_query_template, "r") as f:
        sql_queries_templates = yaml.safe_load(f)

    with open(args.triplets, "r") as f:
        header = f.readline()
        triplets = f.readlines()

    triplets = [t.rstrip("\n").split("\t") for t in triplets]

    # Connect to the AACT database
    db_uri = f"postgresql+psycopg2://{args.user}:{args.pwd}@{HOST}:{PORT}/{DATABASE}"
    sql_db = SQLDatabase.from_uri(db_uri, include_tables=AACT_TABLES)
    sql_schema = [STUDY_TABLE] + [sql_db.get_single_table_info(t) for t in AACT_TABLES]
    sql_schema = "\n".join(sql_schema)
    
    if verbose:
        #dspy_tracing(host="http://0.0.0.0")
        print("SQL db schema:\n" + sql_schema)

    if args.hf:
        os.environ["HUGGING_FACE_TOKEN"] = args.hf
    
    if args.vllm:
        lm = dspy.HFClientVLLM(model=args.vllm, port=8000, url="http://0.0.0.0", max_tokens=1_000, timeout_s=2_000)
        file_tags.append(args.vllm.split("/")[-1])
        
    elif args.ollama:
        lm = dspy.OllamaLocal(
            model=args.ollama, stop=args.stop, max_tokens=1_000, timeout_s=2_000
        )
        file_tags.append(args.ollama)

    dspy.settings.configure(lm=lm, temperature=0.1)
        
    query_engine = Txt2SqlAgent(sql_db, sql_schema, COMMON_MISTAKES)

    sql_eval = run_sql_eval(
        query_engine,
        sql_db,
        sql_queries_templates,
        triplets,
        verbose,
    )
    sql_eval.to_csv(
        f"{args.output_dir}{'.'.join(file_tags)}.eval.tsv",
        sep="\t",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test dspy txt2sql")

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
        "-ollama",
        type=str,
        default="mistral",
        help="Large Language Model name using Ollama nomenclature. Default: 'mistral'.",
    )
    parser.add_argument(
        "-stop", type=str, nargs="+", default=["INST", "/INST"], help=""
    )

    parser.set_defaults(hf=None, vllm=None)

    args = parser.parse_args()
    main(args, verbose=False)
