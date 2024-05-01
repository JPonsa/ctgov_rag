
import os
from neo4j import GraphDatabase

#os.environ["NEO4J_URI"] = "bolt://0.0.0.0:7687"
#os.environ["NEO4J_URI"] = "neo4j://0.0.0.0:7474"
#os.environ["NEO4J_URI"] = "bolt://vital-totally-bison.ngrok-free.app/"
os.environ["NEO4J_URI"] = 'neo4j+s://e00b1042.databases.neo4j.io'
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = '4xoaq5bmjTqWyzKfWBlF8kcEQ5TOvThIv1QhsDeAlFs'
os.environ["NEO4J_DATABASE"] = "ctgov"


driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

print("Got driver")
driver.verify_connectivity()
print("End")
