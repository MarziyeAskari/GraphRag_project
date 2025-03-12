import os
import pandas as pd
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
from neo4j_graphrag.indexes import create_vector_index
from embedding_function import *  # Your function to compute embeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Neo4jGraph:
    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection with credentials from environment variables."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://staging.shamim.review:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "lmis_neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logging.info("Successfully connected to Neo4j.")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def load_csv_to_graph(self, phenotype_to_genes_csv, phenotype_csv):
        """Load nodes and relationships from local CSV files into Neo4j with batch processing."""
        try:
            phenotype_to_genes = pd.read_csv(phenotype_to_genes_csv)
            phenotype = pd.read_csv(phenotype_csv)
        except Exception as e:
            logging.error(f"Error reading CSV files: {e}")
            return

        # Load phenotype_to_genes.csv in batch
        query_1 = """
        UNWIND $rows AS row
        MERGE (s:Symptom {id: row.hpo_id})
        ON CREATE SET s.name = row.hpo_name
        MERGE (g:Gene {id: row.gene_id})
        ON CREATE SET g.name = row.gene_symbol
        MERGE (d:Disease {id: row.disease_id})
        MERGE (s)-[:HAS_GENE]->(g)
        MERGE (s)-[:ASSOCIATED_WITH]->(d)
        MERGE (d)-[:HAS_SYMPTOM]->(s)
        MERGE (g)-[:ASSOCIATED_WITH]->(d)
        MERGE (d)-[:HAS_GENE]->(g)
        MERGE (g)-[:IMPACTS]->(s);
        """
        query_2 = """
        UNWIND $rows AS row
        MERGE (d:Disease {id: row.database_id})
        ON CREATE SET d.name = row.disease_name
        MERGE (s:Symptom {hpo_id: row.hpo_id})
        MERGE (d)-[:HAS_SYMPTOM]->(s);
        """

        with self.driver.session() as session:
            try:
                session.run(query_1, rows=phenotype_to_genes.to_dict("records"))
                session.run(query_2, rows=phenotype.to_dict("records"))
                logging.info("Data successfully loaded into Neo4j.")
            except Exception as e:
                logging.error(f"Error executing Neo4j queries: {e}")

    def update_name_embeddings(self):
        """
        For all nodes labels, compute and update the 'name_embedding'
        property based on the 'name' property.
        """
        with self.driver.session() as session:
            # Get all nodes with a 'name' property
            query_fetch = "MATCH (n:Symptom) WHERE exists(n.name) RETURN n.id AS id, n.name AS name"
            results = session.run(query_fetch)
            for record in results:
                node_id = record["id"]
                name_text = record["name"]
                # Compute the embedding (this function should be implemented to use your embedding model)
                embedding_vector = get_embedding(name_text)
                # Update the node with the computed embedding
                query_update = """
                MATCH (n:Symptom {hpo_id: $node_id})
                SET n.name_embedding = $embedding
                """
                session.run(query_update, node_id=node_id, embedding=embedding_vector)
            logging.info("Name embeddings updated for all Symptom nodes.")

    def update_disease_names_and_embeddings(self, disease_name_mapping):
        """
        Update 'Disease' nodes where 'name' is NULL by assigning a name from a mapping and computing 'name_embedding'.

        :param disease_name_mapping: Dictionary mapping disease IDs to their names.
        """
        with self.driver.session() as session:
            # Fetch Disease nodes with null name
            query_fetch = "MATCH (d:Disease) WHERE d.name IS NULL RETURN d.id AS id"
            results = session.run(query_fetch)

            for record in results:
                disease_id = record["id"]

                # Assign name if available in mapping
                if disease_id in disease_name_mapping:
                    disease_name = disease_name_mapping[disease_id]
                    embedding_vector = get_embedding(disease_name)  # Compute embedding

                    # Update Disease node in Neo4j
                    query_update = """
                    MATCH (d:Disease {id: $disease_id})
                    SET d.name = $disease_name, d.name_embedding = $embedding
                    """
                    session.run(query_update, disease_id=disease_id, disease_name=disease_name,
                                embedding=embedding_vector)

            logging.info("Updated Disease nodes with names and embeddings.")

    def create_vector_index(self, index_name, label, embedding_property,
                            dimensions=1536, similarity_fn="cosine"):
        """
        Create a vector index on nodes with the given label and embedding property.
        """
        try:
            create_vector_index(
                self.driver,
                index_name,
                label=label,
                embedding_property=embedding_property,
                dimensions=dimensions,
                similarity_fn=similarity_fn
            )
            logging.info(
                f"Vector index '{index_name}' created on label '{label}' using property '{embedding_property}'.")
        except Exception as e:
            logging.error(f"Error creating vector index: {e}")


# Example Usage:
if __name__ == "__main__":
    neo4j_db = Neo4jGraph()
    # Load your CSV data into Neo4j
    # neo4j_db.load_csv_to_graph("datasets/phenotype_to_genes.csv", "datasets/phenotype.csv")

    # Update nodes with embeddings computed from the 'name' property
    # neo4j_db.update_name_embeddings()
    # Load disease names from CSV
    # disease_df = pd.read_csv("datasets/phenotype.csv")
    # # disease_df_ = load_disease_name_mapping("datasets/phenotype_to_genes.csv")
    # disease_name_mapping = dict(zip(disease_df["database_id"], disease_df["disease_name"]))

    # Update Disease nodes with names and embeddings
    # neo4j_db.update_disease_names_and_embeddings(disease_name_mapping)

    # Create a vector index on the 'name_embedding' property for semantic search

    neo4j_db.create_vector_index(
        index_name="text_embeddings_symptom",
        label="Symptom",
        embedding_property="name_embedding",
        dimensions=1536,
        similarity_fn="cosine"
    )

    neo4j_db.close()
