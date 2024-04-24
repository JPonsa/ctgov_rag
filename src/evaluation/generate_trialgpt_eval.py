import json

import pandas as pd
from tqdm import tqdm

trialgpt_dir = "./data/raw/trialgpt/"
datasets = {
    "sigir": trialgpt_dir + "sigir/trial_sigir.json",
    "trec_2021": trialgpt_dir + "trec/trial_2021.json",
    "trec_2022": trialgpt_dir + "trec/trial_2022.json",
}

prompt = """Based on the following patient description, provide a list of 5 or less clinical trials ids where this patient would be eligible to be enrolled. Patient note: {patient_note}"""

df = pd.DataFrame(
    [], columns=["id", "dataset", "patient_id", "question", "0", "1", "2"]
)
for dataset_name, file_path in datasets.items():
    with open(file_path, "r") as f:
        dataset = json.load(f)
        for note in tqdm(dataset, desc=dataset_name):
            patient_id = note["patient_id"]
            patient_note = note["patient"]
            id = f"{dataset_name}_{patient_id}"
            studies = {"0": [], "1": [], "2": []}

            for i in studies.keys():
                if i in note.keys():
                    for study in note[i]:
                        studies[i].append(study["NCTID"])

            tmp = pd.DataFrame(
                [
                    [
                        id,
                        dataset_name,
                        patient_id,
                        prompt.format(patient_note=patient_note)
                        .replace("\n\n", "\n")
                        .replace("\n", "|"),
                        studies["0"],
                        studies["1"],
                        studies["2"],
                    ]
                ],
                columns=df.columns,
            )
            df = pd.concat([df, tmp], ignore_index=True)

df.to_csv("./src/evaluation/trialgpt.eval.tsv", sep="\t", index=None)
