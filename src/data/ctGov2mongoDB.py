# ctGov2mongoDB.py ###############################################
#
# Provided with a list of Clinical Trials ids (nctId).
# Query the Clinical Trials API and load them into mongoDB cloud.


import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import connect_to_mongoDB, get_clinical_trial_study


def ETL(studies: list, collection):

    for nct_id in tqdm(studies, desc="Loading studies into Mongo DB"):
        study = get_clinical_trial_study(nct_id)
        # that ID used by mongoDB to study dict
        tmp = {"_id": nct_id}
        tmp.update(study)
        study = tmp
        collection.insert_one(study)


if __name__ == "__main__":

    load_dotenv(".env")
    MONGODB_USER = os.getenv("MONGODB_USER")
    MONGODB_PWD = os.getenv("MONGODB_PWD")

    client = connect_to_mongoDB(MONGODB_USER, MONGODB_PWD)

    db = client["ctGov"]

    for disease in ["heart_failure", "asthma"]:
        # Get list of studies
        acct_file = f"./data/aact.browse_conditions.{disease}.csv"
        df = pd.read_csv(acct_file)
        studies = np.unique(df["nct_id"].values)

        # Extract from ct.gov and Load into mongo db
        collection = db[disease]
        for nct_id in tqdm(studies, desc=f"Loading {disease} studies into Mongo DB"):
            study = get_clinical_trial_study(nct_id)
            # that ID used by mongoDB to study dict
            tmp = {"_id": nct_id}
            tmp.update(study)
            study = tmp
            collection.insert_one(study)

    # close connection to mongo db
    client.close()
