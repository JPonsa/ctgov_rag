import argparse
import json
import os
import sys

os.environ ['CUDA_LAUNCH_BLOCKING'] ='1' # For vLLM error reporting
os.environ["DPS_CACHEBOOL"]='False' # dspy no cache

import dspy
from dspy.retrieve.neo4j_rm import Neo4jRM
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
import pandas as pd
from sentence_transformers import SentenceTransformer


####### Add src folder to the system path so it can call utils
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.sql_wrapper import SQLDatabase
from utils.utils import dspy_tracing, print_red

# dspy_tracing()

VERBOSE = True

# TODO: Remove credentials
# Neo4j credentials
#os.environ["NEO4J_URI"] = "bolt://127.0.0.1:7687"
os.environ["NEO4J_URI"] = 'neo4j+s://e5534dd1.databases.neo4j.io'
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'Jih6YsVFgkmwpbt26r7Lm4dIuFWG8fOnvlXc-2fj9SE'
os.environ["NEO4J_DATABASE"] = 'neo4j'

# AACT credentials
os.environ["AACT_USER"] = "jponsa"
os.environ["AACT_PWD"] = "aact.ctti-clinicaltrials.org"

# Embedding model
biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")


# TODO: Review if this is the best way. Could if be an alternative in which
# chunks are summarised and aggregated into one single contexts? 

def check_context_length(x:str, max_token:int=2_500)->str:
    CHR_PER_TOKEN = 4
    max_characters = max_token*CHR_PER_TOKEN
    if len(x) > max_characters:
        print(f"Context too large {x[:max_characters]} ...")
        return x[:max_characters]+" ..."
    else:
        return x

def str_formatting(x:str, max_token:int=2_500) ->str:
    """Remove some special characters that could be confusing the LLM or interfering with the post processing of the text"""
    
    if not isinstance(x, str):
        print("NOT A STRING!!!!!")
        return x
    
    x = x.replace('"',"")
    x = x.replace("{","")
    x = x.replace("}","")
    x = x.replace("[","")
    x = x.replace("],",";")
    x = x.replace("]","")
    
    x = check_context_length(x, max_token)
    
    #TODO: Assess whether it is required to limite the str lenght 
    # to fix token / character lenght.
    
    if x in ["", " "]:
        x =  "Tool produced no response."
    
    return x

def fromToCt_query(from_node: str, from_property: str, ct_properties: list[str]) -> str:

    ct_properties_str = ", ".join([f'{p} = "+ct.{p}+" ' for p in ct_properties]) + '"'

    query = """
    WITH node, score
    OPTIONAL MATCH (node)-[:{from_node}ToClinicalTrialAssociation]->(ct:ClinicalTrial)
    WITH node, ct, max(score) AS score // deduplicate parents
    RETURN "{from_node}: "+node.{from_property}+". ClinicalTrial: {ct_properties_str} AS text, score, {{}} AS metadata
    """
    cmd = query.format(
        from_node=from_node,
        from_property=from_property,
        ct_properties_str=ct_properties_str,
    )
    return cmd


def fromToCtTo_query(
    from_node: str, from_property: str, to_node: str, to_property: str
) -> str:

    query = """
    WITH node, score
    OPTIONAL MATCH path = (node)-[:{from_node}ToClinicalTrialAssociation]->(ct:ClinicalTrial)-[:ClinicalTrialTo{to_node}Association]->(target:{to_node})
    WITH node.{from_property} AS from_node_txt, COLLECT(DISTINCT target.{to_property}) AS to_node_list, max(score) AS score // deduplicate parents
    RETURN "{from_node}: "+from_node_txt+". {to_node}: "+apoc.text.join(to_node_list, ', ') AS text, score, {{}} AS metadata
    """
    cmd = query.format(
        from_node=from_node,
        from_property=from_property,
        to_node=to_node,
        to_property=to_property,
    )
    return cmd


def get_sql_engine(model:str, model_host:str, model_port:int):
    from llama_index.core import Settings, SQLDatabase
    from llama_index.core.query_engine import NLSQLTableQueryEngine
    from sqlalchemy import create_engine

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

    from llama_index.llms.openai_like import OpenAILike
    sql_lm = OpenAILike(model=model, api_base=f"{model_host}:{model_port}/v1/", api_key="fake", temperature=0, max_tokens=1_000)  
    Settings.llm = sql_lm
    Settings.embed_model = "local"

    user = os.getenv("AACT_USER")
    pwd = os.getenv("AACT_PWD")
    # TODO: Review if this it the right way to hardcode this.
    AACT_DATABASE = "aact"
    AACT_HOST = "aact-db.ctti-clinicaltrials.org"
    AACT_PORT = 5432
    
    uri = f"postgresql+psycopg2://{user}:{pwd}@{AACT_HOST}:{AACT_PORT}/{AACT_DATABASE}"
    db_engine = create_engine(uri)
    sql_db = SQLDatabase(db_engine, include_tables=TABLES)
    query_engine = NLSQLTableQueryEngine(sql_database=sql_db)
    return query_engine

from  ReAct import (
    get_cypher_engine, 
    cypher_engine,
    get_sql_engine, 
    ChitChat, 
    GetClinicalTrial, 
    ClinicalTrialToEligibility, 
    InterventionToCt, 
    InterventionToAdverseEvent, 
    ConditionToCt, 
    ConditionToIntervention, 
    AnalyticalQuery,
    MedicalSME
    )

class PatientEligibility(dspy.Signature):
    "Given a patient description, produce a list of 5 or less clinical trials ids where tha patient would be eligible for enrolment."
    patient_note:str = dspy.InputField(prefix="Patient Note:", desc="description of the patient medical characteristics and conditions")
    ct_ids:str = dspy.OutputField(prefix="Clinical Trials ids:", desc="a comma separated list of clinical trials e.g. NCT0001,NCT0002,NCT0003")

def main(args):
    k=5
    KG_tools = [
    GetClinicalTrial(),
    ClinicalTrialToEligibility(),
    InterventionToCt(k=k),
    InterventionToAdverseEvent(k=k),
    ConditionToCt(k=k),
    ConditionToIntervention(k=k),
    ]

    tools = [ChitChat()]
    
    
    #---- Define the tools to be used
    valid_methods = ["sql_only", "kg_only","cypher_only", "llm_only", "analytical_only", "all"]
    if args.method not in valid_methods:
        raise NotImplementedError(f"method={args.method} not supported. methods must be one of {valid_methods}")
    
    if args.method == "sql_only":
        tools += [AnalyticalQuery(args, sql=True, kg=False)]
    
    elif args.method == "kg_only":
        tools += [AnalyticalQuery(args, sql=False, kg=True)]
        tools += KG_tools
    
    elif args.method == "cypher_only":
        tools += [AnalyticalQuery(args, sql=False, kg=True)]
    
    elif args.method == "llm_only":
        pass
    
    elif args.method == "analytical_only":
        tools += [AnalyticalQuery(args, sql=True, kg=True)]
        
    else:
        tools += [AnalyticalQuery(args, sql=True, kg=True)]
        tools += KG_tools
        
    if args.med_sme:
        # TODO: Not hardcoded or better set.
        sme_model = "TheBloke/meditron-7B-GPTQ"
        sme_host = "http://0.0.0.0"
        sme_port = 8051
        tools += [MedicalSME(sme_model, sme_host, sme_port)]
    
    react_module = dspy.ReAct(PatientEligibility, tools=tools, max_iters=3)
    
    
    #---- Load the LLM
    lm = dspy.HFClientVLLM(model=args.vllm, port=args.port, url=args.host, max_tokens=1_000, timeout_s=2_000, 
                           stop=['\n\n', '<|eot_id|>'], 
                        #    model_type='chat',
                           )
    dspy.settings.configure(lm=lm, temperature=0.3)
    
    #---- Get questioner
    questioner = pd.read_csv(args.input_tsv, sep="\t", index_col=0)
    
    #---- Answer questioner
    for idx, row in questioner.iterrows():
        patient_note = row.patient_note
        print("#####################")
        print(f"Question: {patient_note}")
        result = react_module(patient_note=patient_note)
        questioner.loc[idx, "ReAct_answer"] = result.ct_ids
        print(f"Final Predicted Answer (after ReAct process): {result.ct_ids}")
        
    #---- Save response
    questioner.to_csv(args.output_tsv, sep="\t", index=None)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="ct.gov ReAct")
    
    parser.add_argument(
        "-vllm",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
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
        default=8_000,
        help="LLM server port.",
    )
    
        # Add arguments
    parser.add_argument(
        "-i",
        "--input_tsv",
        type=str,
        default="./data/ctGov.questioner.mistral7b.tsv",
        help="path to questioner file. It assumes that the file is tab-separated. that the file contains 1st column as index and a `question` column.",
    )

    # Add arguments
    parser.add_argument(
        "-o",
        "--output_tsv",
        type=str,
        default="./results/ReAct/ctGov.questioner.mistral7b.tsv",
        help="full path to the output tsv file. The file will contain the same information as the input file plus an additional `ReAct_answer` column.",
    )
    
        # Add arguments
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="all",
        help="""inference methods`sql_only`, `kg_only`, `cypher_oly`, `all`.
        `sql_only` user txt-2-SQL llamaindex tool directly to AACT. 
        `kg_only` uses a set of pre-defined tools for Vector Search and txt-2-Cypher on a Neo4j KG.
        `cypher_only` uses txt-2-Cypher LnagChian tool on a Neo4j KG.
        `all` user all tools available.
        Default `all`."""
    )
    parser.add_argument("-s","--med_sme", action='store_true', help="Flag indicating the access to a Med SME LLM like Meditron. Default: False")
    
    
    parser.set_defaults(vllm=None, med_sme=False)

    args = parser.parse_args()
   
    main(args)
    print("ReAct - Completed")