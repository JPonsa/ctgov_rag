import json
import os
import sys

import dspy
from dspy.retrieve.neo4j_rm import Neo4jRM
from neo4j import GraphDatabase
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

dspy_tracing()

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
PORT=8042
HOST="http://0.0.0.0"

# TODO: Remove credentials
# Neo4j credentials
os.environ["NEO4J_URI"] = "bolt://127.0.0.1:7687"
#os.environ["NEO4J_URI"] = "bolt://0.0.0.0:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["NEO4J_DATABASE"] = "ctgov"

# AACT credentials
os.environ["AACT_USER"] = "jponsa"
os.environ["AACT_PWD"] = "aact.ctti-clinicaltrials.org"

# Embedding model
biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")


def fromToCt_query(from_node: str, from_property: str, ct_properties: list[str]) -> str:

    ct_properties_str = ", ".join([f'{p} = "+ct.{p}+" ' for p in ct_properties]) + '"'

    query = """
    WITH node, score
    OPTIONAL MATCH (node)-[:{from_node}ToStudyAssociation]->(ct:ClinicalTrial)
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
    OPTIONAL MATCH path = (node)-[:{from_node}ToStudyAssociation]->(ct:ClinicalTrial)-[:StudyTo{to_node}Association]->(target:{to_node})
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


def get_cypher_engine(model:str, model_host:str, model_port:int):
    from langchain.chains import GraphCypherQAChain

    # TODO: Remove unnecessary import
    # from langchain.graphs import Neo4jGraph
    from langchain_community.graphs import Neo4jGraph
    
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PWD")

    from langchain_community.llms import VLLMOpenAI
    
    cypher_lm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=f"{model_host}:{model_port}/v1/",
        model_name=model,
        # model_kwargs={"stop": ["."]},
        )

    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=user,
        password=pwd,
        database="ctgov",
    )

    chain = GraphCypherQAChain.from_llm(cypher_lm, graph=graph, verbose=True)
    return chain


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField()


class QAwithContext(dspy.Signature):
    """Given a question and context provide an answer."""

    question: str = dspy.InputField(prefix="Question:", desc="question to be answered.")
    sql_response: str = dspy.InputField(
        prefix="SQL response",
        desc="contains information that could be relevant to the question.",
    )
    cypher_response: str = dspy.InputField(
        prefix="Cypher response:",
        desc="contains information that could be relevant to the question.",
    )
    answer: str = dspy.OutputField(
        prefix="Answer:", desc="final response to the question."
    )


class ChitChat(dspy.Module):
    """Provide a question to a generic answer"""

    name = "ChitChat"
    input_variable = "question"
    desc = "Simple question that does not require specialised knowledge and you know the answer."

    def __init__(self):
        self.chatter = dspy.Predict(BasicQA)

    def __call__(self, question):
        return self.chatter(question=question).answer


class GetClinicalTrial(dspy.Module):
    """Retrieve Summary of a Clinical Trial"""

    name = "GetClinicalTrial"
    input_variable = "nctid_list"
    desc = "Given a list of clinical trials ids (e.g. ['NCT00000173', 'NCT00000292']) get Clinical Trials summaries."

    def __init__(self):

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def __call__(self, nctid_list: list) -> str:
        fields = ["id", "brief_title", "study_type", "keywords", "brief_summary"]
        query = "MATCH (ClinicalTrial:ClinicalTrial) WHERE ClinicalTrial.id IN [{nctid_list}] RETURN {field_list}".format(
            nctid_list=",".join(["'" + x + "'" for x in nctid_list]),
            field_list=", ".join("ClinicalTrial." + f for f in fields),
        )

        response, _, _ = self.driver.execute_query(
            query, database_=os.getenv("NEO4J_DATABASE")
        )
        text = {}
        for r in response:
            text[r[f"ClinicalTrial.id"]] = {
                f: r[f"ClinicalTrial.{f}"] for f in fields if f != "id"
            }

        text = json.dumps(text)
        return text


class ClinicalTrialToEligibility(dspy.Module):
    """Retrieve a Clinical Trial eligibility criteria"""

    name = "ClinicalTrialToEligibility"
    input_variable = "nctid_list"
    desc = "Given a list of clinical trials ids (e.g. ['NCT00000173', 'NCT00000292']) get the trial eligibility criteria."

    def __init__(self):

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def __call__(self, nctid_list: list) -> str:
        fields = [
            "healthy_volunteers",
            "minimum_age",
            "maximum_age",
            "sex",
            "eligibility_criteria",
        ]

        query = """MATCH (ct:ClinicalTrial)-[:StudyToEligibilityAssociation]->(e:Eligibility)
        WHERE ct.id IN [{nctid_list}] RETURN ct.id, {field_list}
        """.format(
            nctid_list=",".join(["'" + x + "'" for x in nctid_list]),
            field_list=", ".join("e." + f for f in fields),
        )

        response, _, _ = self.driver.execute_query(
            query, database_=os.getenv("NEO4J_DATABASE")
        )
        text = {}
        for r in response:
            text[r[f"ct.id"]] = {f: r[f"e.{f}"] for f in fields}


class InterventionToCt(dspy.Module):
    name = "InterventionToCt"
    input_variable = "intervention"
    desc = "Get the Clinical Trials associated to a medical Intervention."

    def __init__(self):
        self.retriever = Neo4jRM(
            index_name="intervention_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=10,
            retrieval_query=fromToCt_query(
                "Intervention", "name", ["id", "study_type", "brief_title"]
            ),
        )
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str, k: int = 5) -> str:
        print(f"InterventionToCt({intervention})")
        response = self.retriever(intervention, k)
        response = "\n".join([x["long_text"] for x in response])
        print(response)
        return response


class InterventionToAdverseEvent(dspy.Module):
    name = "InterventionToAdverseEvent"
    input_variable = "intervention"
    desc = "Get the Adverse Events associated to a medical Intervention tested in a clinical trial."

    def __init__(self):
        self.retriever = Neo4jRM(
            index_name="intervention_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=10,
            retrieval_query=fromToCtTo_query(
                "Intervention", "name", "AdverseEvent", "term"
            ),
        )
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str, k: int = 5) -> str:
        print(f"InterventionToAdverseEvent({intervention})")
        response = self.retriever(intervention, k)
        response = "\n".join([x["long_text"] for x in response])
        return response


class ConditionToCt(dspy.Module):
    name = "ConditionToCt"
    input_variable = "condition"
    desc = "Get the Clinical Trials associated to a medical condition."

    def __init__(self):
        self.retriever = Neo4jRM(
            index_name="condition_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=10,
            retrieval_query=fromToCt_query(
                "Condition", "name", ["id", "study_type", "brief_title"]
            ),
        )
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str, k: int = 5) -> str:
        print(f"InterventionToCt({intervention})")
        response = self.retriever(intervention, k)
        response = "\n".join([x["long_text"] for x in response])
        print(response)
        return response


class ConditionToIntervention(dspy.Module):
    name = "ConditionToIntervention"
    input_variable = "condition"
    desc = "Get the medical Interventions associated to a medical Condition tested in a clinical trial."

    def __init__(self):
        self.retriever = Neo4jRM(
            index_name="condition_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=10,
            retrieval_query=fromToCtTo_query(
                "Condition", "name", "Intervention", "name"
            ),
        )
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str, k: int = 5) -> str:
        print(f"ConditionToIntervention({intervention})")
        response = self.retriever(intervention, k)
        response = "\n".join([x["long_text"] for x in response])
        return response


class ConditionToIntervention(dspy.Module):
    name = "ConditionToIntervention"
    input_variable = "condition"
    desc = "Get the medical Interventions associated to a medical Condition tested in a clinical trial."

    def __init__(self):
        self.retriever = Neo4jRM(
            index_name="condition_biobert_emb",
            text_node_property="name",
            embedding_provider="huggingface",
            embedding_model="dmis-lab/biobert-base-cased-v1.1",
            k=10,
            retrieval_query=fromToCtTo_query(
                "Condition", "name", "Intervention", "name"
            ),
        )
        self.retriever.embedder = biobert.encode

    def __call__(self, intervention: str, k: int = 5) -> str:
        print(f"ConditionToIntervention({intervention})")
        response = self.retriever(intervention, k)
        response = "\n".join([x["long_text"] for x in response])
        return response


class MedicalSME(dspy.Module):
    name = "MedicalSME"
    input_variable = "question"
    desc = ""

    def __init__(self, model:str, host:str, port:int):
        self.SME = dspy.ChainOfThought(BasicQA)    
        self.lm = dspy.HFClientVLLM(model=model, port=port, url=host, max_tokens=1_000, timeout_s=2_000)

    def __call__(self, question) -> str:
        with dspy.context(lm=self.lm, temperature=0.7):
            response = self.lm(question).answer
        return response


class AnalyticalQuery(dspy.Module):
    name = "AnalyticalQuery"
    input_variable = "question"
    desc = "Access to a db of clinical trials. Reply question that could be answered with a SQL query. Use when other tools are not suitable."

    def __init__(self, sql:bool=True, kg:bool=True):
        sql_engine = get_sql_engine(model=MODEL, model_host=HOST, model_port=PORT)
        cypher_engine = get_cypher_engine(model=MODEL, model_host=HOST, model_port=PORT)
        response_generator = dspy.Predict(QAwithContext)
        self.sql = sql
        self.kg = kg
        if not (self.sql or self.kg):
            raise ValueError("Either SQL query or KG query must be enabled by setting it to True.")

    def __call__(self, question):
        
        sql_response=""
        cypher_response=""
        
        if self.sql:
            try:
                sql_response = self.sql_engine.query(question)
            except Exception as e:
                sql_response = "Sorry, I could not provide an answer."
                
        if self.kg:
            try:
                cypher_response = self.cypher_engine.invoke(question)
            except Exception as e:
                cypher_response = "Sorry, I could not provide an answer."

        response = self.response_generator(
            question=question,
            sql_response=sql_response,
            cypher_response=cypher_response,
        ).answer

        return response


def main(questions:list, method:str="all", med_sme:bool=True):
    
    KG_tools = [
    GetClinicalTrial(),
    ClinicalTrialToEligibility(),
    InterventionToCt(),
    InterventionToAdverseEvent(),
    ConditionToCt(),
    ConditionToIntervention(),
    ]

    tools = [ChitChat()]
    
    valid_methods = ["sql-only", "lg-only", "all"]
    if method not in valid_methods:
        raise NotImplementedError(f"method={method} not supported. methods must be one of {valid_methods}")
    
    if method == "sql-only":
        tools += [AnalyticalQuery(sql=True, kg=False)]
    
    elif method == "kg-only":
        tools += [AnalyticalQuery(sql=False, kg=True)]
        tools += KG_tools
        
    else:
        tools += [AnalyticalQuery(sql=True, kg=True)]
        tools += KG_tools
        
    if med_sme:
        # TODO: Not hardcoded or better set.
        sme_model = "TheBloke/meditron-7B-GPTQ"
        sme_host = "http://0.0.0.0"
        sme_port = 8051
        tools += [MedicalSME(sme_model, sme_host, sme_port)]
    
    react_module = dspy.ReAct(BasicQA, tools=tools, max_iters=3)
    
    lm = dspy.HFClientVLLM(model=MODEL, port=PORT, url=HOST, max_tokens=1_000, timeout_s=2_000)
    dspy.settings.configure(lm=lm, temperature=0.3)
    

    for question in questions:
        result = react_module(question=question)
        print(f"Question: {question}")
        print(f"Final Predicted Answer (after ReAct process): {result.answer}")


if __name__ == "__main__":
    
    questions = [
        "What are the adverse events associated with drug Acetaminophen?",
        "What intervention is studies in clinical trial NCT00000173?",
        "How many people where individual enrolled in clinical trial NCT00000173?",
        "Why the sky is blue?",
    ]

    main(questions, method="all", med_sme=False)