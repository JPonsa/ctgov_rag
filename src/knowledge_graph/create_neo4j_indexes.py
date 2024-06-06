import argparse
import os

from neo4j import GraphDatabase


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

    # NEO4J_USER = "tester"
    # NEO4J_PWD = "password"

    NEO4J_USER = "neo4j"
    # NEO4J_PWD = "password"
    # URI = "bolt://localhost:7687"

    NEO4J_URI = "neo4j+s://e5534dd1.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "Jih6YsVFgkmwpbt26r7Lm4dIuFWG8fOnvlXc-2fj9SE"

    AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    # DB_NAME = "ctgov"
    DB_NAME = "neo4j"

    with GraphDatabase.driver(NEO4J_URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        
        # Index CT node based on id
        cmd = "CREATE INDEX FOR (n:ClinicalTrial) ON (n.id)"
        driver.execute_query(cmd, database_=DB_NAME)

        # Clinical Trials
        create_emb_index(
            driver,
            DB_NAME,
            "ct_trial2vec_emb",
            "ClinicalTrial",
            "trial2vec_emb",
            128,
        )
        
        create_emb_index(
            driver,
            DB_NAME,
            "ct_biobert_emb",
            "ClinicalTrials",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            DB_NAME,
            "ct_kw",
            "ClinicalTrials",
            ["keywords"],
        )

        # Condition
        create_emb_index(
            driver,
            DB_NAME,
            "condition_biobert_emb",
            "Condition",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            DB_NAME,
            "condition_kw",
            "Condition",
            ["name"],
        )

        # Intervention
        create_emb_index(
            driver,
            DB_NAME,
            "intervention_biobert_emb",
            "Intervention",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            DB_NAME,
            "intervention_kw",
            "Intervention",
            ["name"],
        )

        # Outcome
        create_emb_index(
            driver,
            DB_NAME,
            "outcome_biobert_emb",
            "Outcome",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            DB_NAME,
            "outcome_kw",
            "Outcome",
            ["measure"],
        )

        # # Biospec
        # create_emb_index(
        #     driver,
        #     DB_NAME,
        #     "biospec_biobert_emb",
        #     "Biospec",
        #     "biobert_emb",
        #     768,
        # )

        # create_kwd_index(
        #     driver,
        #     DB_NAME,
        #     "biospec_kw",
        #     "Biospec",
        #     ["description"],
        # )

        # Adverse Event
        create_emb_index(
            driver,
            DB_NAME,
            "ae_biobert_emb",
            "AdverseEvent",
            "biobert_emb",
            768,
        )
        create_kwd_index(
            driver,
            DB_NAME,
            "ae_kw",
            "AdverseEvent",
            ["term"],
        )

        # OrganSystem
        create_emb_index(
            driver,
            DB_NAME,
            "og_biobert_emb",
            "OrganSystem",
            "biobert_emb",
            768,
        )

        create_kwd_index(
            driver,
            DB_NAME,
            "og_kw",
            "OrganSystem",
            ["name"],
        )
