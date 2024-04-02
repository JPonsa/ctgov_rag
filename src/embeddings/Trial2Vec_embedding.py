import pandas as pd
from tqdm import tqdm


def ct_dict2pd(study: dict, missing_val=None) -> pd.Series:
    """ETL process to convert a CT study in a JSON format to the format
    exemplified in the trial2vec demo data.
    See: https://pypi.org/project/Trial2Vec/

    Parameters
    ----------
    study : dict
        as provided through the clinicaltrials.gov API

    missing_val: (default:None)
        How to encode missing values

    Returns
    -------
    pd.Series
        fields :
        - nct_id
        - description
        - study_type
        - title
        - intervention name
        - disease
        - keyword
        - outcome measure
        - (selection) criteria
        - references
        - overall status
    """
    missing_val = None

    ct_protocol = study.get("protocolSection", {})

    nct_id = ct_protocol.get("identificationModule", {}).get("nctId", missing_val)

    description = ct_protocol.get("descriptionModule", {}).get(
        "briefSummary", missing_val
    )

    study_type = ct_protocol.get("designModule", {}).get("studyType", missing_val)

    title = ct_protocol["identificationModule"].get(
        "officialTitle",
        ct_protocol["identificationModule"].get("briefTitle", missing_val),
    )

    # Intervention name
    if study_type == "OBSERVATIONAL":
        intervention_name = study_type
    else:
        interventions = ct_protocol.get("armsInterventionsModule", {}).get(
            "interventions", []
        )
        intervention_name = ", ".join(
            set(i.get("name", "").split(":")[-1] for i in interventions)
        )

    disease = ", ".join(
        sorted(ct_protocol.get("conditionsModule", {}).get("conditions", []))
    )

    keyword = (
        ", ".join(sorted(ct_protocol.get("conditionsModule", {}).get("keywords", [])))
        if study_type != "OBSERVATIONAL"
        else missing_val
    )

    # Outcome measurement
    if study_type == "OBSERVATIONAL":
        try:
            design_info = ct_protocol["designModule"]["designInfo"]
            outcome_measure = design_info.get("observationalModel", study_type)
            outcome_measure += (
                "-" + design_info.get("timePerspective", "")
                if "timePerspective" in design_info
                else ""
            )
        except KeyError:
            outcome_measure = study_type
    else:
        primary_outcomes = ct_protocol.get("outcomesModule", {}).get(
            "primaryOutcomes", []
        )
        outcome_measure = ", ".join(set(i.get("measure", "") for i in primary_outcomes))

    # Selection criteria
    try:
        criteria = ct_protocol.get("eligibilityModule", {}).get(
            "eligibilityCriteria", ""
        )
        criteria = criteria.replace("\n* ", "~").replace("\n", "~").replace("~~", "~")
    except:
        try:
            eligibility = ct_protocol.get("eligibilityModule", {})
            criteria = ", ".join(
                [": ".join([k, str(v)]) for k, v in eligibility.items()]
            )
        except:
            criteria = missing_val

    # References
    try:
        references = ct_protocol.get("referencesModule", {}).get("references", [])
        tmp = []
        for r in references:
            try:
                tmp.append(r["citation"].split(".")[1].lstrip(" "))
            except IndexError:
                pass
        reference = ", ".join(tmp)
    except KeyError:
        reference = missing_val

    overall_status = (
        ct_protocol.get("statusModule", {}).get("overallStatus", "").lower()
    )

    return (
        pd.Series(
            {
                "nct_id": nct_id,
                "description": description,
                "title": title,
                "intervention_name": intervention_name,
                "disease": disease,
                "keyword": keyword,
                "outcome_measure": outcome_measure,
                "criteria": criteria,
                "reference": reference,
                "overall_status": overall_status,
            }
        )
        .to_frame()
        .transpose()
    )


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from trial2vec import Trial2Vec

    from utils import connect_to_mongoDB

    load_dotenv(".env")
    MONGODB_USER = os.getenv("MONGODB_USER")
    MONGODB_PWD = os.getenv("MONGODB_PWD")

    model = Trial2Vec(device="cpu")
    model.from_pretrained()

    client = connect_to_mongoDB(MONGODB_USER, MONGODB_PWD)
    db = client["ctGov"]

    for disease in ["heart_failure", "asthma"]:
        study_pd = pd.DataFrame()
        collection = db[disease]
        studies = collection.find({})

        for study in tqdm(studies, desc=f"Reformating {disease} CT studies"):
            tmp = ct_dict2pd(study)

            study_pd = pd.concat([study_pd, tmp])

        print(f"Embedding {disease} studies using trial2vec")
        emb = model.encode({"x": study_pd})
        emb_file = f"./data/ct.trial2vec_embedding.{disease}.csv"
        pd.DataFrame(emb).to_csv(emb_file, index=False)
        print(f"Storing embeddings into {emb_file}")

        collection.update_many({}, "")
        for study in tmp(emb.keys(), desc="Storing trial2vec embeddings into mongo db"):
            collection.update_one({"_id": study}, {"$set": {"trial2vec": emb[study]}})
