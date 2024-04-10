# Download datasets from TrialGPT
# Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10418514/

import os
import urllib.request

import pandas as pd

trialgpt_git = "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/datasets/"

trialgpt = "./data/raw/trialgpt/"
if not os.path.exists(trialgpt):
    os.makedirs(trialgpt)

sigir = trialgpt + "sigir/"
if not os.path.exists(sigir):
    os.makedirs(sigir)

trec = trialgpt + "trec/"
if not os.path.exists(trec):
    os.makedirs(trec)


# Download trial_2021.json if it doesn't exist
if not os.path.exists(trec + "trial_2021.json"):
    urllib.request.urlretrieve(
        trialgpt_git + "trial_2021.json", trec + "trial_2021.json"
    )

# Download trial_2022.json if it doesn't exist
if not os.path.exists(trec + "trial_2022.json"):
    urllib.request.urlretrieve(
        trialgpt_git + "trial_2022.json", trec + "trial_2022.json"
    )

# Download trial_sigir.json if it doesn't exist
if not os.path.exists(sigir + "trial_sigir.json"):
    urllib.request.urlretrieve(
        trialgpt_git + "trial_sigir.json", sigir + "trial_sigir.json"
    )

# Download trial2info.json if it doesn't exist
if not os.path.exists(trialgpt + "trial2info.json"):
    urllib.request.urlretrieve(
        trialgpt_git + "trial2info.json", trialgpt + "trial2info.json"
    )

    df = pd.read_json(trialgpt + "trial2info.json")

    df = df.head().T
    df.index.names = ["nct_id"]
    df.to_csv("./data/trialgtp.studies_list.csv")
