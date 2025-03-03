from neo4j import GraphDatabase
from get_API_key import get_API_key
import os
import logging
from neo4j_database import Neo4jGraph

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API Key securely
os.environ["OPENAI_API_KEY"] = get_API_key()

# Initialize the Neo4j Graph driver
driver = Neo4jGraph().driver

# Query to Find Related Genes
QUERY = """
MATCH (d:Disease)
WHERE d.name CONTAINS $disease_name
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)<-[:ASSOCIATED_WITH]-(g:Gene)
RETURN d.name AS disease, collect(DISTINCT s.name) AS symptoms, collect(DISTINCT g.name) AS genes
"""

def get_related_genes(disease_name):
    with driver.session() as session:
        result = session.run(QUERY, {"disease_name": disease_name})
        for record in result:
            return record["symptoms"]  # Access the 'genes' field

# Find Genes for "Premature centromere division"
disease_name = "Premature centromere division"
symptoms = get_related_genes(disease_name)

# Print Results
print(f"Genes related to '{disease_name}': {symptoms}")
