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


def trim_collection(collection, file_path: str):
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
            collection.update_many({}, {"$unset": {idx: 1}})

    collection.update_many({}, {"$unset": {"trial2vec": 1}})


def main(
    user: str,
    pwd: str,
    db_name: str,
    collections: list[str],
    file_path: str,
    overwrite: bool = False,
) -> None:

    with connect_to_mongoDB(user, pwd) as client:
        db = client[db_name]

        if overwrite:  # Trim the existing collection
            for c in collections.split(","):
                collection = db[c]
                trim_collection(collection, file_path)
                print(f"Trimming {c} collection - done")

        else:  # Create a copy called preprocessed and trim it.
            preprocessed = db["preprocessed"]
            preprocessed.delete_many({})
            print("Clearing preprocessed collection -  done")

            # merge all collections into one
            for c in collections.split(","):
                print(f"Coping {c} collection into preprocessed ...")
                collection = db[c]
                preprocessed.with_options(write_concern=WriteConcern(w=0)).insert_many(
                    collection.find({}), ordered=False
                )

            print("Copying to preprocessed collection - done")
            trim_collection(preprocessed, file_path)
            print("Trimming preprocessed collection - done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trims aMongo DB cloud collection based on metadata"
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
    parser.add_argument(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Set the overwrite value to True. If True, the trimming happens directly in the collection. Otherwise, creates a copy called 'preprocessed' and  applies the trimming there.",
    )
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()
    user = args.user
    pwd = args.pwd
    db_name = args.database
    collections = args.collections
    file_path = args.metadata
    overwrite = args.overwrite

    main(user, pwd, db_name, collections, file_path, overwrite)
