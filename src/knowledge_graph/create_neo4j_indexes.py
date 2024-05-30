import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv("./.env")


def delete_index(driver, db: str, idx_name: str) -> None:
    """Delete Neo4j index

    Parameters
    ----------
    driver : Neo4j driver
    db : str
        database name
    idx_name : str
        index name
    """

    query = f"DROP INDEX {idx_name} IF EXISTS"
    driver.execute_query(query, database_=db)


def create_emb_index(
    driver,
    db: str,
    idx_name: str,
    node_label: str,
    emb_node_property: str,
    vector_dim: int,
    similarity_func: str = "cosine",
    overwrite: bool = True,
    verbose: bool = False,
) -> None:
    """Create (embedding) vector index

    Parameters
    ----------
    driver : Neo4j driver
    db : str
        database name
    idx_name : str
        index name
    node_label : str
        node Label in Neo4j
    emb_node_property : str
        name of the note property that contains the vector index
    vector_dim: int,
        length of the embedding vector
    similarity_func : str
        metric to measure similarity. Either "cosine" or "euclidean", default "cosine"
    overwrite: bool,
        if True it deletes the index before creating it, default True
    verbose: bool,
        if True prints the cypher query used to create the index, default False

    See: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
    """

    query = """CREATE VECTOR INDEX `{idx_name}` IF NOT EXISTS
    FOR (n:{node_label})
    ON (n.{emb_node_property})
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {vector_dim},
        `vector.similarity_function` : '{similarity_func}'
        }}}}
    """
    if overwrite:
        delete_index(driver, db, idx_name)

    cmd = query.format(
        idx_name=idx_name,
        node_label=node_label,
        emb_node_property=emb_node_property,
        vector_dim=vector_dim,
        similarity_func=similarity_func,
    )

    if verbose:
        print(f"Creating index {idx_name} ...")
        print(cmd)

    driver.execute_query(cmd, database_=db)


def create_kwd_index(
    driver,
    db: str,
    idx_name: str,
    node_label: str,
    node_properties: list[str],
    overwrite: bool = True,
    verbose: bool = False,
) -> None:
    """Create keyword index

    Parameters
    ----------
    driver : Neo4j driver
    db : str
        database name
    idx_name : str
        index name
    node_label : str
        node Label in Neo4j
    node_properties : list[str]
        list of node properties use to perform the keyword search
    overwrite: bool,
        if True it deletes the index before creating it, default True
    verbose: bool,
        if True prints the cypher query used to create the index, default False

    See: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/
    """

    node_properties_list = ", ".join(["n." + x for x in node_properties])

    query = """CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS
    FOR (n:{node_label}) 
    ON EACH [{node_properties_list}]
    """

    if overwrite:
        delete_index(driver, db, idx_name)

    cmd = query.format(
        idx_name=idx_name,
        node_label=node_label,
        node_properties_list=node_properties_list,
    )

    if verbose:
        print(f"Creating index {idx_name} ...")
        print(cmd)

    driver.execute_query(cmd, database_=db)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create Neo4j Node Vector Indexes and Node Keywords indexes"
    )

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

    with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
        driver.verify_connectivity()

        # Clinical Trials
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "ct_trial2vec_emb",
            "ClinicalTrial",
            "trial2vec_emb",
            128,
        )

        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "ct_biobert_emb",
            "ClinicalTrials",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "ct_kw",
            "ClinicalTrials",
            ["keywords"],
        )

        # Condition
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "condition_biobert_emb",
            "Condition",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "condition_kw",
            "Condition",
            ["name"],
        )

        # Intervention
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "intervention_biobert_emb",
            "Intervention",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "intervention_kw",
            "Intervention",
            ["name"],
        )

        # Outcome
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "outcome_biobert_emb",
            "Outcome",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "outcome_kw",
            "Outcome",
            ["measure"],
        )

        # # Biospec
        # create_emb_index(
        #     driver,
        #     NEO4J_DATABASE,
        #     "biospec_biobert_emb",
        #     "Biospec",
        #     "biobert_emb",
        #     768,
        # )

        # create_kwd_index(
        #     driver,
        #     NEO4J_DATABASE,
        #     "biospec_kw",
        #     "Biospec",
        #     ["description"],
        # )

        # Adverse Event
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "ae_biobert_emb",
            "AdverseEvent",
            "biobert_emb",
            768,
        )
        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "ae_kw",
            "AdverseEvent",
            ["term"],
        )

        # OrganSystem
        create_emb_index(
            driver,
            NEO4J_DATABASE,
            "og_biobert_emb",
            "OrganSystem",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            NEO4J_DATABASE,
            "og_kw",
            "OrganSystem",
            ["name"],
        )

print("Index Creation  - Done")
