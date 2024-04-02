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

FORMATTER = {"float": lambda x: "%.10f" % x}
SEPARATOR = "|"
PRECISION = 10


def trial2vect_encode_node(inp_path, out_path):

    studies = []
    study_pd = pd.DataFrame()

    node_name = "ClinicalTrials"
    header = pd.read_csv(f"{inp_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{inp_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    for nctId in df[":ID"].values:
        studies.append(get_clinical_trial_study(nctId))

    for study in studies:
        tmp = ct_dict2pd(study)
        study_pd = pd.concat([study_pd, tmp])

    embeddings = trial2vec.encode({"x": study_pd})

    for i, row in tqdm(df.iterrows(), desc=f"Embedding {node_name}"):

        nctId = row[":ID"]
        emb = embeddings[nctId]
        emb_str = (
            np.array2string(
                emb, separator=SEPARATOR, precision=PRECISION, formatter=FORMATTER
            )
            .replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )

        df.at[i, "trial2vec_emb:double[]"] = emb_str

    df.to_csv(f"{out_path}{node_name}-part000.csv", sep="\t", header=None, index=False)


def biobert_enconde_node(inp_path, out_path, node_name, term):

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
        bio_emb = bio_emb[0]
        bio_emb = (
            np.array2string(
                bio_emb, separator=SEPARATOR, precision=PRECISION, formatter=FORMATTER
            )
            .replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        df.at[i, "biobert_emb:double[]"] = bio_emb

    df.to_csv(f"{out_path}{node_name}-part000.csv", sep="\t", header=None, index=False)


def main():

    trial2vec = Trial2Vec(device=DEVICE)
    trial2vec.from_pretrained()
    biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")

    inp_path = "./data/raw/knowledge_graph/"
    out_path = "./data/preprocessed/knowledge_graph/"

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
    main()
