import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from trial2vec import Trial2Vec

from src.embeddings.embeddings import BioBert2Vect
from src.utils.utils import get_clinical_trial_study

device = "cuda:0" if torch.cuda.is_available() else "cpu"

trial2vec_model = Trial2Vec(device=device)
trial2vec_model.from_pretrained()

biobert2vect = BioBert2Vect()


def enconde_node(inp_path, out_path, node_name, term, prefix: str = ""):

    header = pd.read_csv(f"{inp_path}{node_name}-header.csv", sep="\t")
    df = pd.read_csv(f"{inp_path}{node_name}-part000.csv", sep="\t", header=None)
    df.columns = header.columns

    df["tria2vec_emb:double[]"] = df["tria2vec_emb:double[]"].astype(object)
    df["biobert_emb:double[]"] = df["biobert_emb:double[]"].astype(object)

    formatter = {"float": lambda x: "%.10f" % x}
    separator = "|"
    precision = 10

    for i, row in tqdm(df.iterrows(), desc=f"Embedding {node_name}"):

        if i > 10:  # TODO: keep it to top 10 for testing
            break

        sentence = prefix + row[term]

        t2v = trial2vec_model.sentence_vector(sentence)
        t2v = t2v[0].numpy()
        t2v = (
            np.array2string(
                t2v, separator=separator, precision=precision, formatter=formatter
            )
            .replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        df.at[i, "tria2vec_emb:double[]"] = t2v

        bio = biobert2vect.get_sentence_embedding(sentence, method="last_hidden_state")
        bio = bio[0]
        bio = (
            np.array2string(
                bio, separator=separator, precision=precision, formatter=formatter
            )
            .replace("\n", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )
        df.at[i, "biobert_emb:double[]"] = bio

    df.to_csv(f"{out_path}{node_name}-part000.csv", sep="\t", header=None, index=False)


def main():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial2vec_model = Trial2Vec(device=device)
    trial2vec_model.from_pretrained()

    inp_path = "./data/raw/knowledge_graph/"
    out_path = "./data/preprocessed/knowledge_graph/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    enconde_node(inp_path, out_path, "AdverseEvent", "term", "Adverse Event: ")
    enconde_node(inp_path, out_path, "Biospec", "description")
    enconde_node(inp_path, out_path, "Condition", ":ID", "Condition: ")
    enconde_node(inp_path, out_path, "Eligibility", "eligibility_criteria")
    enconde_node(inp_path, out_path, "Intervention", ":ID")
    enconde_node(inp_path, out_path, "Outcome", "measure")


if __name__ == "__main__":
    main()
