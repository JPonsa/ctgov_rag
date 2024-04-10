# ctGov2mongoDB.py ###############################################
#
# Provided with a list of Clinical Trials ids (nctId).
# Query the Clinical Trials API and load them into mongoDB cloud.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pymongo
from pymongo.write_concern import WriteConcern
from tqdm import tqdm

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import connect_to_mongoDB, get_clinical_trial_study


def ETL(studies: list, collection) -> None:
    """Given a list of clinical trials nct id codes, it pulls them from the ct.gov API
    and copies them into a Mongo DB collection

    Parameters
    ----------
    studies : list
        list of nct ids. Based on the id the study protocol is pulled using the ct.gov API
    collection : Mongo DB collection
        where to store the studies in Mongo DB
    """

    for nct_id in tqdm(studies, desc="Loading studies into Mongo DB"):
        study = get_clinical_trial_study(nct_id)
        # that ID used by mongoDB to study dict
        tmp = {"_id": nct_id}
        tmp.update(study)
        study = tmp
        collection.with_options(write_concern=WriteConcern(w=0)).insert_one(study)


def main(user: str, pwd: str, db_name: str, collection_name: str, file_path: str):
    """Does:
    - Given the user credentials it connects to Mongo DB
    - Given a list of clinical trial nct ids stored in a csv, pulls the studies from ct.gov
    - Copies "like-for-like" the a study from ct.gov into Mongo DB

    Parameters
    ----------
    user : str
        MONGO DB user name
    pwd : str
        MONGO DB password
    db_name : str
        MONGO DB database name
    collection_name : str
        MONGO DB collection name
    file_path : str
        Path to csv file containing the list of nct ids. Assumes the files has column with header "nct_id"
    """

    with connect_to_mongoDB(user, pwd) as client:
        db = client[db_name]
        df = pd.read_csv(file_path)
        studies = np.unique(df["nct_id"].values)
        collection = db[collection_name]
        ETL(studies, collection)


if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser(
        description="Copy a list of studies from clinicaltrials.gov into Mongo DB cloud"
    )
    parser.add_argument("-u", "--user", type=str, help="MONGODB user name.")
    parser.add_argument("-p", "--pwd", type=str, help="MONGODB password.")
    parser.add_argument(
        "-d", "--database", type=str, help="MONGODB database name", default="ctgov"
    )
    parser.add_argument("-c", "--collection", type=str, help="MONGODB collection name")
    parser.add_argument("-s", "--studies", type=str, help="path to the list of studies")

    args = parser.parse_args()
    user = args.user
    pwd = args.pwd
    db_name = args.database
    collection_name = args.collection
    file_path = args.studies

    main(user, pwd, db_name, collection_name, file_path)
