import json

import requests
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def get_clinical_trial_study(nct_id: str) -> dict:
    """Given an nct id get the clinical trials data from clinicaltrials.gov api"

    Parameters
    ----------
    nct_id : str
        clinical trials identifier

    Returns
    -------
    dict
        clinical trials json
    """
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    headers = {"accept": "text/csv"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print("Request failed with status code:", response.status_code)


def connect_to_mongoDB(user: str, pwd: str, app_name: str = "cluster0") -> MongoClient:

    uri = f"mongodb+srv://{user}:{pwd}@{app_name}.bcn2gwy.mongodb.net/?retryWrites=true&w=majority&appName={app_name.capitalize()}"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi("1"), connectTimeoutMS=100_000)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)


def print_red(text):
    """Print a text message in Red"""
    print("\033[91m" + text + "\033[0m")


def print_green(text):
    """Print a text message in Green"""
    print("\033[92m" + text + "\033[0m")
