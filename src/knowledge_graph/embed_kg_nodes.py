import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from trial2vec import Trial2Vec

####### Add src folder to the system path so it can call utils
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from embeddings.Trial2Vec_embedding import ct_dict2pd
from utils.utils import (
    connect_to_mongoDB,
    get_clinical_trial_study,
    print_green,
    print_red,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEPARATOR = "|"
PRECISION = 15
# FORMATTER = {"float": lambda x: f"%.{PRECISION}f" % x}

trial2vec = Trial2Vec(device=DEVICE)
trial2vec.from_pretrained()
biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")


def embedding_formatting(emb: np.array) -> str:
    """Takes a embedding vector and reformats it to a string"""

    emb = (
        np.array2string(emb, separator=SEPARATOR, precision=PRECISION)
        .replace("\n", "")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
    )
    return emb


def trial2vect_encode_node(input_path: str, output_path: str, args):
    """Apply trial2vec to the ClinicalTrail KG nodes

    Parameters
    ----------
    input_path : str
        input data path
    output_path : str
        output data path
    """

    studies = []
    study_pd = pd.DataFrame()

    node_name = "ClinicalTrial"
    header = pd.read_csv(f"{input_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{input_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    df["trial2vec_emb:double[]"] = df["trial2vec_emb:double[]"].astype(object)

    if args.mongoDB:  # Pull studies from mongoDB collection
        with connect_to_mongoDB(user=args.user, pwd=args.pwd) as client:
            db = client[args.db]
            collection = db[args.collection]
            print(
                f"Downloading studies from mongoDB db:{args.db} collection:{args.collection} ..."
            )
            studies = [doc for doc in collection.find({})]

    else:  # Pull studies from ct.gov API
        print("Downloading studies from ct.gov ...")
        for nctId in df[":ID"].values:
            studies.append(get_clinical_trial_study(nctId))

    print("Reshaping studies to be embedded using trial2vec")
    for study in studies:
        tmp = ct_dict2pd(study)
        study_pd = pd.concat([study_pd, tmp])

    embeddings = trial2vec.encode({"x": study_pd})

    print(f"Embedding {node_name}")
    for i, row in df.iterrows():

        nctId = row[":ID"]
        emb = embeddings[nctId]
        emb_str = embedding_formatting(emb)

        df.at[i, "trial2vec_emb:double[]"] = emb_str

    df.to_csv(
        f"{output_path}{node_name}-part000.csv", sep="\t", header=None, index=False
    )


def biobert_encode_node(input_path: str, output_path: str, node_name: str, term: str):
    """Apply bioBERT embeddings to a KG node"

    Parameters
    ----------
    input_path : str
        input data path
    output_path : str
        output data path
    node_name : str
        Name of the KG note that appears in the file name.
        {input_path}{node_name}-header.csv
    term : str
        fields in the file corresponding to the node attribute to be encoded
    """

    header = pd.read_csv(f"{input_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{input_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    if "biobert_emb:double[]" not in df.columns:
        return None

    df["biobert_emb:double[]"] = df["biobert_emb:double[]"].astype(object)

    print(f"Embedding {node_name}")
    for i, row in df.iterrows():

        sentence = row[term]

        # Convert sentence to biobert embedding
        bio_emb = biobert.encode(sentence)
        bio_emb_str = embedding_formatting(bio_emb)
        df.at[i, "biobert_emb:double[]"] = bio_emb_str

    df.to_csv(
        f"{output_path}{node_name}-part000.csv", sep="\t", header=None, index=False
    )


def main(args):

    if os.path.exists(args.output_path):
        print_red(f"WARNING! {args.output_path} already exist! in will be overwritten!")
        shutil.rmtree(args.output_path)

    # Copy the entire input directory into the output directory
    shutil.copytree(args.input_path, args.output_path)

    trial2vect_encode_node(args.output_path, args.output_path, args)
    biobert_encode_node(
        args.output_path, args.output_path, "ClinicalTrial", "brief_title"
    )
    biobert_encode_node(args.output_path, args.output_path, "AdverseEvent", "term")
    # biobert_encode_node(args.output_path, args.output_path, "Biospec", "description")
    biobert_encode_node(args.output_path, args.output_path, "Condition", "name")
    biobert_encode_node(args.output_path, args.output_path, "Intervention", "name")
    biobert_encode_node(args.output_path, args.output_path, "Outcome", "measure")
    biobert_encode_node(args.output_path, args.output_path, "OrganSystem", "name")

    # update neo4j-admin command
    neo4j_admin = "neo4j-admin-import-call-windows.sh"

    with open(args.input_path + neo4j_admin, "r") as f:
        command = f.readline()

    windows_input_path = args.input_path.replace("./", "/").replace("/", ("\\"))
    windows_output_path = args.output_path.replace("./", "/").replace("/", ("\\"))
    command = command.replace(windows_input_path, windows_output_path)

    with open(args.output_path + neo4j_admin, "w") as f:
        command = f.write(command)

    print_green(f"Knowledge Graph Node Embedding COMPLETED. See {args.output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="embed clinical trials KG")
    parser.add_argument("input_path", type=str, help="Input path.")
    parser.add_argument("output_path", type=str, help="Output path.")
    parser.add_argument(
        "-mongoDB",
        dest="mongoDB",
        action="store_true",
        help="Set mongoDB flag to True. If True,  get the CT studies from mongoDB otherwise from ct.gov",
    )
    parser.add_argument("-user", type=str, help="MongoDB user name")
    parser.add_argument("-pwd", type=str, help="MongoDB password")
    parser.add_argument("-app", type=str, default="cluster0", help="MongoDB cluster")
    parser.add_argument("-db", type=str, default="ctGov", help="MongoDB database name")
    parser.add_argument("-c", "--collection", type=str, help="MongoDB collection name")
    parser.set_defaults(mongoDB=False)

    args = parser.parse_args()
    main(args)
