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


from utils.utils import connect_to_mongoDB


def main(
    user: str, pwd: str, db_name: str, collections: list[str], file_path: str
) -> None:

    with connect_to_mongoDB(user, pwd) as client:
        db = client[db_name]

        preprocessed = db["preprocessed"]
        preprocessed.delete_many({})

        # merge all collections into one
        for c in collections.split(","):
            collection = db[c]
            preprocessed.with_options(write_concern=WriteConcern(w=0)).insert_many(
                collection.find({}), ordered=False
            )

        # remove unnecessary / unwanted fields
        for sheet in [
            "protocolSection",
            "resultsSection",
            "annotationSection",
            "documentSection",
            "derivedSection",
        ]:
            df = pd.read_excel(file_path, sheet)

            for idx in df.loc[df["Used"] == "N", "Index Field"].values:
                preprocessed.update_many({}, {"$unset": {idx: 1}})

        preprocessed.update_many({}, {"$unset": {"trial2vec": 1}})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Copy a list of studies from clinicaltrials.gov into Mongo DB cloud"
    )
    parser.add_argument("-u", "--user", type=str, help="MONGODB user name.")
    parser.add_argument("-p", "--pwd", type=str, help="MONGODB password.")
    parser.add_argument(
        "-d", "--database", type=str, help="MONGODB database name", default="ctgov"
    )
    parser.add_argument(
        "-c", "--collections", type=str, help="comma separated list of collection"
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        help="path to studies metadata file",
        default="./docs/ctGov.metadata.xlsx",
    )

    args = parser.parse_args()
    user = args.user
    pwd = args.pwd
    db_name = args.database
    collections = args.collections
    file_path = args.metadata

    main(user, pwd, db_name, collections, file_path)
