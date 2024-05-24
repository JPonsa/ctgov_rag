import argparse
import json
import os
import sys

os.environ ['CUDA_LAUNCH_BLOCKING'] ='1' # For vLLM error reporting
os.environ["DPS_CACHEBOOL"]='False' # dspy no cache

import dspy
from dspy.retrieve.neo4j_rm import Neo4jRM
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
import pandas as pd
from sentence_transformers import SentenceTransformer

from  ReAct import (
    str_formatting,
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



def list_parser(x:str) -> list:
    """Takes a comma-separated list as a string and returns a list of words"""
    
    if not isinstance(x, str):
        return []
    
    return x.replace("[", "").replace("]","").replace("'", "").replace('"',"").replace(" ", "").split(",")


def precision(example:dspy.Example, prediction:dspy.Prediction, trace=None)->float:
    """Computes the precision based on 2 comma-separated lists (as str)"""
    
    # Transform to list of words
    example = list_parser(example.clinical_trial_ids_list)
    prediction = list_parser(prediction.clinical_trial_ids_list)
    
    # list -> set
    example = set(example)
    prediction = set(prediction)
    
    # compute prediction precision. Proportion of words in prediction appear in the example list
    score = len(prediction.intersection(example))/len(prediction)
    
    print(f"Precision : {score}")
    
    return score
    
class PatientEligibility(dspy.Signature):
    "Given a patient description, produce a list of 5 or less clinical trials ids where tha patient would be eligible for enrolment."
    patient_note:str = dspy.InputField(prefix="Patient Note:", desc="description of the patient medical characteristics and conditions")
    clinical_trial_ids_list:str = dspy.OutputField(prefix="Clinical Trials ids:", desc="a comma-separated list of clinical trials e.g. NCT0001,NCT0002,NCT0003")

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

    # tools = [ChitChat()]
    tools= []
    
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
        tools = [ChitChat()]
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
    
    
    class ReActPipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            self.signature = PatientEligibility
            self.predictor = dspy.ReAct(self.signature, tools=tools, max_iters=3)
    
        def forward(self, patient_note):
            return self.predictor(patient_note=patient_note) 
    
    # react_module = dspy.ReAct(PatientEligibility, tools=tools, max_iters=3)
    
    #---- Load the LLM
    lm = dspy.HFClientVLLM(model=args.vllm, port=args.port, url=args.host, max_tokens=1_000, timeout_s=2_000, 
                           stop=['\n\n', '<|eot_id|>'], 
                        #    model_type='chat',
                           )
    dspy.settings.configure(lm=lm, temperature=0.3)
    
    #---- Get questioner
    questioner = pd.read_csv(args.input_tsv, sep="\t", index_col=0)
    
    # Train / Test split
    training, evaluation = questioner.iloc[:100, :].copy(), questioner.iloc[100:,:].copy()
    
    # Create input and output examples
    trainset = []
    for i, row in training.iterrows():
        trainset.append(dspy.Example(patient_note=row["patient_note"], clinical_trial_ids_list=row["2"]).with_inputs("patient_note"))


    devset = []
    for i, row in evaluation.iterrows():
        devset.append(dspy.Example(patient_note=row["patient_note"], clinical_trial_ids_list=row["2"]).with_inputs("patient_note"))
        
        
    #---- Evaluation
    evaluate_program = Evaluate(devset=devset, metric=precision, num_threads=2, display_progress=True, display_table=5)
    print("---- Evaluation starting ReAct pipeline ----")
    evaluate_program(ReActPipeline())
    
    #---- Training
    # config = dict(max_bootstrapped_demos=3, max_labeled_demos=3, num_candidate_programs=10, num_threads=4)
    # teleprompter = BootstrapFewShotWithRandomSearch(metric=precision, **config)
    # optimized_program = teleprompter.compile(ReActPipeline(), trainset=trainset, valset=devset)
    # optimized_program.save(f"./models/trialGPT.React{args.method}.json")
    
    # print("---- Evaluation optimised ReAct pipeline ----")
    # evaluate_program(optimized_program)
    
    optimized_program = ReActPipeline()
    
    evaluation["ReAct_answer"]= "" # Set output field
    
    for idx, row in evaluation.iterrows():
        patient_note = row.patient_note
        print("#####################")
        print(f"Question: {patient_note}")
        # result = react_module(patient_note=patient_note)
        result = optimized_program(patient_note=patient_note)
        evaluation.loc[idx, "ReAct_answer"] = result.clinical_trial_ids_list
        print(f"Final Predicted Answer (after ReAct process): {result.clinical_trial_ids_list}")
        
    #---- Save response
    evaluation.to_csv(args.output_tsv, sep="\t", index=None)

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