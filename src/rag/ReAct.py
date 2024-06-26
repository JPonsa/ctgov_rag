import argparse
import json
import os
import sys

os.environ ['CUDA_LAUNCH_BLOCKING'] ='1' # For vLLM error reporting
os.environ["DPS_CACHEBOOL"]='False' # dspy no cache

import dspy
from dspy.retrieve.neo4j_rm import Neo4jRM
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
import pandas as pd
from sentence_transformers import SentenceTransformer

import spacy

load_dotenv('./.env')

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

def str_formatting(x:str, max_token:int=2_500, rm_stop:bool=False) ->str:
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
    
    if rm_stop:
        # trim the text removing stop tokens
        nlp = spacy.load("en_core_web_sm")
        x = nlp(x)
        x = [token.text for token in x if not token.is_stop]
        x = " ".join(x)
    
    # x = check_context_length(x, max_token)
    
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
    sql_lm = OpenAILike(model=model, api_base=f"{model_host}:{model_port}/v1/", api_key="fake", 
                        temperature=0, max_tokens=1_000, 
                        is_chat_model=True,
                        )  
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


def get_cypher_engine(model:str, model_host:str, model_port:int):
    from langchain.chains import GraphCypherQAChain

    # TODO: Remove unnecessary import
    # from langchain.graphs import Neo4jGraph
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.llms import VLLMOpenAI
    
    cypher_lm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=f"{model_host}:{model_port}/v1/",
        model_name=model,
        # model_kwargs={"stop": ["."]},
        )

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )

    chain = GraphCypherQAChain.from_llm(cypher_lm, graph=graph, verbose=False, validate_cypher=True)
    return chain


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField()


class QAwithContext(dspy.Signature):
    """Given a question and context return an answer."""

    question: str = dspy.InputField(prefix="Question:", desc="question to be answered.")
    sql_response: str = dspy.InputField(
        prefix="SQL response:",
        desc="contains information that could be relevant to the question.",
    )
    cypher_response: str = dspy.InputField(
        prefix="Cypher response:",
        desc="contains information that could be relevant to the question.",
    )
    answer: str = dspy.OutputField(
        prefix="Answer:", desc="final response to the question."
    )


class Txt2Cypher(dspy.Signature):
    """"Takes an input question and a Knowledge Graph db schema to produce a syntactically correct cypher query to run.
    Pay attention to the relationships between nodes and their directionality"""
    question:str = dspy.InputField(prefix="Question:")
    graph_schema:str = dspy.InputField(prefix="Knowledge Graph schema:")
    cypher_query:str = dspy.OutputField(prefix="Cypher query:")
    
    
def cypher_engine(question:str)->str:
    """Takes a question and returns the result of a Cypher query"""
    
    from langchain_community.graphs import Neo4jGraph
    
    graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    )
    
    response = "Sorry. I could not find this information"
    cypher_query_eng = dspy.Predict(Txt2Cypher)
    #BUG:LLM is adding a double-quotation at the end and sometimes at the begging.
    cypher_query = cypher_query_eng(question=question, graph_schema=graph.schema).cypher_query.strip('"').strip("---")
    
    if cypher_query:
        print(f"Cypher query: {cypher_query}")
        
        response = graph.query(cypher_query)
        if isinstance(response, list):
            response = "|".join(response)
    
    return response
    

class ChitChat(dspy.Module):
    """Provide a response to a generic question"""

    name = "ChitChat"
    input_variable = "question"
    desc = "Simple question that does not require specialised knowledge and you know the answer."

    def __init__(self):
        self.chatter = dspy.Predict(BasicQA)

    def __call__(self, question):
        if VERBOSE:
            print(f"Action: ChitChat({question})")
        
        return self.chatter(question=question).answer


class GetClinicalTrial(dspy.Module):
    """Retrieve Summary of a Clinical Trial"""

    name = "GetClinicalTrial"
    input_variable = "clinical_trial_ids_list"
    desc = "Takes a comma separated list of clinical trials ids and gets Clinical Trials summaries."
    # desc = "Given a comma separated list of clinical trials ids (e.g. NCT00000173,NCT00000292) get Clinical Trials summaries."

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")
        self.db = os.getenv("NEO4J_DATABASE")

    def __call__(self, clinical_trial_ids_list: list) -> str:
        if VERBOSE:
            print(f"Action: GetClinicalTrial({clinical_trial_ids_list})")
        
        fields = ["id", "brief_title", "study_type", "keywords", "brief_summary"]
        query = "MATCH (ct:ClinicalTrial) WHERE ct.id IN [{clinical_trial_ids_list}] RETURN {field_list}".format(
            clinical_trial_ids_list=",".join(["'" + x.strip(" ") + "'" for x in clinical_trial_ids_list.split(",")]),
            field_list=", ".join("ct." + f for f in fields),
        )
        
        with GraphDatabase.driver(self.uri, auth=(self.user, self.pwd)) as driver:
            neo4j_response, _, _ = driver.execute_query(query, database_=self.db)
        
        text = {}
        for r in neo4j_response:
            text[r[f"ClinicalTrial.id"]] = {
                f: r[f"ClinicalTrial.{f}"] for f in fields if f != "id"
            }

        response = str_formatting(json.dumps(text))
        
        if VERBOSE:
            print(f"Function Response: {response}")
        
        return response


class ClinicalTrialToEligibility(dspy.Module):
    """Retrieve a Clinical Trial eligibility criteria"""

    name = "ClinicalTrialToEligibility"
    input_variable = "clinical_trial_ids_list"
    desc = "Takes a comma separated list of clinical trials ids and gets the trial eligibility criteria."
    # desc = "Given a comma separated list of clinical trials ids (e.g. NCT00000173,NCT00000292) get the trial eligibility criteria."

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.pwd = os.getenv("NEO4J_PASSWORD")
        self.db = os.getenv("NEO4J_DATABASE")

    def __call__(self, clinical_trial_ids_list: list) -> str:
        
        if VERBOSE:
            print(f"Action: ClinicalTrialToEligibility({clinical_trial_ids_list})")
        
        fields = [
            "healthy_volunteers",
            "minimum_age",
            "maximum_age",
            "sex",
            "eligibility_criteria",
        ]

        query = """MATCH (ct:ClinicalTrial)-[:ClinicalTrialToEligibilityAssociation]->(e:Eligibility)
        WHERE ct.id IN [{clinical_trial_ids_list}] RETURN ct.id, {field_list}
        """.format(
            clinical_trial_ids_list=",".join(["'" + x.strip(" ") + "'" for x in clinical_trial_ids_list.split(",")]),
            field_list=", ".join("e." + f for f in fields),
        )

        with GraphDatabase.driver(self.uri, auth=(self.user, self.pwd)) as driver:
            neo4j_response, _, _ = driver.execute_query(query, database_=self.db)
        
        text = {}
        for r in neo4j_response:
            text[r[f"ct.id"]] = {f: r[f"e.{f}"] for f in fields}

        response = str_formatting(json.dumps(text), max_token=1_500)
        
        if VERBOSE:
            print(f"Function Response: {response}")
        
        return response

class InterventionToCt(dspy.Module):
    name = "InterventionToCt"
    input_variable = "intervention"
    desc = "Retrieve the Clinical Trials associated to a medical Intervention."

    def __init__(self, k:int=5):
        self.retriever = Neo4jRM(
            index_name="intervention_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=k,
            retrieval_query=fromToCt_query(
                "Intervention", "name", ["id", "study_type", "brief_title"]
            ),
        )
        self.k = k
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str) -> str:
        
        if VERBOSE:
            print(f"Action: InterventionToCt({intervention})")
        
        response = self.retriever(intervention, self.k) or "Tool produced no response."
        response = str_formatting("\n".join([x["long_text"] for x in response]))
        
        if VERBOSE:
            print(f"Function Response: {response}")
        
        return response


class InterventionToAdverseEvent(dspy.Module):
    name = "InterventionToAdverseEvent"
    input_variable = "intervention"
    desc = "Retrieve the Adverse Events associated to a medical Intervention tested in a Clinical Trial."

    def __init__(self, k:int=5):
        self.retriever = Neo4jRM(
            index_name="intervention_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=k,
            retrieval_query=fromToCtTo_query(
                "Intervention", "name", "AdverseEvent", "term"
            ),
        )
        self.k = k
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str) -> str:
        
        if VERBOSE:
            print(f"Action: InterventionToAdverseEvent({intervention})")
    
        response = self.retriever(intervention, self.k) or "Tool produced no response."
        response = str_formatting("\n".join([x["long_text"] for x in response]))
        
        if VERBOSE:
            print(response)
        
        return response


class ConditionToCt(dspy.Module):
    name = "ConditionToCt"
    input_variable = "condition"
    desc = "Retrieve the Clinical Trials associated to a medical Condition."

    def __init__(self, k:int=5):
        self.retriever = Neo4jRM(
            index_name="condition_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=k,
            retrieval_query=fromToCt_query(
                "Condition", "name", ["id", "study_type", "brief_title"]
            ),
        )
        self.k = k
        self.retriever.embedder = biobert.encode

    def __call__(self, condition: str) -> str:
        
        if VERBOSE:
            print(f"Action: ConditionToCt({condition})")

        response = self.retriever(condition, self.k) or "Tool produced no response."
        response = str_formatting("\n".join([x["long_text"] for x in response]))
        
        if VERBOSE:
            print(response)
        
        return response


class ConditionToIntervention(dspy.Module):
    name = "ConditionToIntervention"
    input_variable = "condition"
    desc = "Retrieve the medical Interventions associated to a medical Condition tested in a Clinical Trial."

    def __init__(self, k:int=5):
        self.retriever = Neo4jRM(
            index_name="condition_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=k,
            retrieval_query=fromToCtTo_query(
                "Condition", "name", "Intervention", "name"
            ),
        )
        self.k = k
        self.retriever.embedder = biobert.encode

    def __call__(self, condition: str) -> str:
        
        if VERBOSE:
            print(f"Action: ConditionToIntervention({condition})")

        response = self.retriever(condition, self.k) or "Tool produced no response."
        response = str_formatting("\n".join([x["long_text"] for x in response]))
        
        if VERBOSE:
            print(f"Function Response: {response}")
        
        return response

class MedicalSME(dspy.Module):
    name = "MedicalSME"
    input_variable = "question"
    desc = ""

    def __init__(self, model:str, host:str, port:int):
        self.SME = dspy.ChainOfThought(BasicQA)    
        self.lm = dspy.HFClientVLLM(model=model, port=port, url=host, max_tokens=1_000, timeout_s=2_000, 
                                    stop=['\n\n', '<|eot_id|>'],
                                    # model_type='chat',
                                    )
        

    def __call__(self, question) -> str:
        with dspy.context(lm=self.lm, temperature=0.7):
            response = self.lm(question).answer
            
        if VERBOSE:
            print(f"Function Response: {response}")
            
        return response


class AnalyticalQuery(dspy.Module):
    name = "AnalyticalQuery"
    input_variable = "question"
    desc = "Access to a db of Clinical Trials. Reply question that could be answered with a SQL query. Use when other tools are not suitable."

    def __init__(self, args, sql:bool=True, kg:bool=True):
        self.sql_engine = get_sql_engine(model=args.vllm, model_host=args.host, model_port=args.port)
        self.cypher_engine = get_cypher_engine(model=args.vllm, model_host=args.host, model_port=args.port)
        self.response_generator = dspy.Predict(QAwithContext)
        self.sql = sql
        self.kg = kg
        if not (self.sql or self.kg):
            raise ValueError("Either SQL query or KG query must be enabled by setting it to True.")

    def __call__(self, question):
        
        if VERBOSE:
            print(f"Action: AnalyticalQuery({question})")
                    
        sql_response=""
        cypher_response=""
        
        if self.sql:
            try:
                sql_response = self.sql_engine.query(question).response
            except Exception as e:
                sql_response = "Sorry, I could not provide an answer."
                
        if self.kg:
            try:
                # BUG: Cypher query making the entire processes to fail. Unknown cause. Taking too long or failing to produce an answer and proceed.
                # raise NotImplementedError("Cypher query making the entire processes to fail. Unknown cause. Taking too long or failing to produce an answer and proceed.")
                cypher_response = cypher_engine(question)
                # cypher_response = self.cypher_engine.invoke(question)["result"]
            except Exception as e:     
                print(f"Cypher query error: {e}")
                cypher_response = "Sorry, I could not provide an answer."
                # cypher_response = "Not implemented."
                
                
        if VERBOSE:
            if self.sql:
                print(f"SQL Response: {sql_response}")

            if self.kg:
                print(f"Cypher Response: {cypher_response}")
                
        response = self.response_generator(
            question=question,
            sql_response=str_formatting(sql_response, max_token=1_000),
            cypher_response=str_formatting(cypher_response, max_token=1_000),
        ).answer

        if VERBOSE:
            print(f"Function Response: {response}")

        return response

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
    
    react_module = dspy.ReAct(BasicQA, tools=tools, max_iters=10)
    
    
    #---- Load the LLM
    lm = dspy.HFClientVLLM(model=args.vllm, port=args.port, url=args.host, max_tokens=1_000, timeout_s=2_000, 
                           stop=['\n\n', '<|eot_id|>'], 
                        #    model_type='chat',
                           )
    dspy.settings.configure(lm=lm, temperature=0.3)
    
    #---- Get questioner
    questioner = pd.read_csv(args.input_tsv, sep="\t", index_col=None)
    questioner["ReAct_answer"]= "" # Set output field
    
    #---- Answer questioner
    for idx, row in questioner.iterrows():
        question = row.question
        print("#####################")
        print(f"Question: {question}")
        result = react_module(question=question)
        questioner.loc[idx, "ReAct_answer"] = result.answer
        print(f"Final Predicted Answer (after ReAct process): {result.answer}")
        
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