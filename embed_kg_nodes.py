import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from trial2vec import Trial2Vec

from src.embeddings.Trial2Vec_embedding import ct_dict2pd
from src.utils.utils import get_clinical_trial_study, print_green, print_red

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")
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


def trial2vect_encode_node(inp_path: str, out_path: str):
    """Apply trial2vec to the ClinicalTrail KG nodes

    Parameters
    ----------
    inp_path : str
        input data path
    out_path : str
        output data path
    """

    studies = []
    study_pd = pd.DataFrame()

    node_name = "ClinicalTrial"
    header = pd.read_csv(f"{inp_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{inp_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    df["trial2vec_emb:double[]"] = df["trial2vec_emb:double[]"].astype(object)

    for i, nctId in enumerate(df[":ID"].values):
        studies.append(get_clinical_trial_study(nctId))

    for i, study in enumerate(studies):
        tmp = ct_dict2pd(study)
        study_pd = pd.concat([study_pd, tmp])

    embeddings = trial2vec.encode({"x": study_pd})

    for i, row in tqdm(df.iterrows(), desc=f"Embedding {node_name}"):

        nctId = row[":ID"]
        emb = embeddings[nctId]
        emb_str = embedding_formatting(emb)

        df.at[i, "trial2vec_emb:double[]"] = emb_str

    df.to_csv(f"{out_path}{node_name}-part000.csv", sep="\t", header=None, index=False)


def biobert_enconde_node(inp_path: str, out_path: str, node_name: str, term: str):
    """Apply bioBERT embeddings to a KG node"

    Parameters
    ----------
    inp_path : str
        input data path
    out_path : str
        output data path
    node_name : str
        Name of the KG note that appears in the file name.
        {inp_path}{node_name}-header.csv
    term : str
        fields in the file corresponding to the node attribute to be encoded
    """

    header = pd.read_csv(f"{inp_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{inp_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    if "biobert_emb:double[]" not in df.columns:
        return None

    df["biobert_emb:double[]"] = df["biobert_emb:double[]"].astype(object)

    for i, row in tqdm(df.iterrows(), desc=f"Embedding {node_name}"):

        sentence = row[term]

        # Convert sentence to biobert embedding
        bio_emb = biobert.encode(sentence)
        bio_emb_str = embedding_formatting(bio_emb)
        df.at[i, "biobert_emb:double[]"] = bio_emb_str

    df.to_csv(f"{out_path}{node_name}-part000.csv", sep="\t", header=None, index=False)


def main(inp_path: str, out_path: str):

    if os.path.exists(out_path):
        print_red(f"WARNING! {out_path} already exist! in will be overwritten!")
        shutil.rmtree(out_path)

    # Copy the entire input directory into the output directory
    shutil.copytree(inp_path, out_path)

    trial2vect_encode_node(out_path, out_path)
    biobert_enconde_node(out_path, out_path, "ClinicalTrial", "brief_title")
    biobert_enconde_node(out_path, out_path, "AdverseEvent", "term")
    biobert_enconde_node(out_path, out_path, "Biospec", "description")
    biobert_enconde_node(out_path, out_path, "Condition", "name")
    biobert_enconde_node(out_path, out_path, "Intervention", "name")
    biobert_enconde_node(out_path, out_path, "Outcome", "measure")
    biobert_enconde_node(out_path, out_path, "OrganSystem", "name")

    # update neo4j-admin command
    neo4j_admin = "neo4j-admin-import-call-windows.sh"

    with open(inp_path + neo4j_admin, "r") as f:
        command = f.readline()

    windows_inp_path = inp_path.replace("./", "/").replace("/", ("\\"))
    windows_out_path = out_path.replace("./", "/").replace("/", ("\\"))
    command = command.replace(windows_inp_path, windows_out_path)

    with open(out_path + neo4j_admin, "w") as f:
        command = f.write(command)

    print_green(f"Knowledge Graph Node Embedding COMPLETED. See {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test llamaindex txt2sql")
    parser.add_argument("input_path", type=str, help="Input path.")
    parser.add_argument("output_path", type=str, help="Output path.")
    args = parser.parse_args()
    inp_path = "./data/raw/knowledge_graph/"
    out_path = "./data/preprocessed/knowledge_graph/"
    main(args.input_path, args.output_path)
