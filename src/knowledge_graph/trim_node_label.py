import os

import pandas as pd

directory = "./data/raw/knowledge_graph/"

files = os.listdir(directory)
nodes = []

labels = [
    "AdverseEvent",
    "AdverseEventProtocol",
    "ClinicalTrial",
    "Condition",
    "Eligibility",
    "Intervention",
    "InterventionProtocol",
    "ObservationProtocol",
    "OrganSystem",
    "Outcome",
]


for file in files:
    if file.endswith("header.csv"):
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, sep="\t")
        df.head()
        if ":LABEL" in df.columns:
            node = file.split("-header.csv")[0]
            file_path = os.path.join(directory, node + "-part000.csv")
            print(file_path)
            df = pd.read_csv(file_path, sep="\t", header=None)
            label = df.iloc[0, -1]
            trimmed_label = list(set(label.split("|")).intersection(labels))[0]
            df[df.columns[-1]] = trimmed_label
            df.to_csv(
                file_path,
                sep="\t",
                header=None,
                index=None,
            )
