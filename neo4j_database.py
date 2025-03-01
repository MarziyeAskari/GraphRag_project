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
        self.uri = "bolt://staging.shamim.review:7687"
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

    def update_name_embeddings_by_label(self):
        """
        For each label in the database, compute and update the 'name_embedding'
        property for nodes that have a 'name' property.
        """
        with self.driver.session() as session:
            # Retrieve all labels in the database
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record["label"] for record in labels_result]
            logging.info(f"Found labels: {labels}")

            for label in labels:
                # Fetch nodes for each label that have a 'name' property
                query_fetch = f"MATCH (n:{label}) WHERE n.name IS NOT NULL RETURN n.id as nodeId, n.name as name"
                results = session.run(query_fetch)
                for record in results:
                    try:
                        node_id = record["nodeId"]
                        name_text = record["name"]
                        # Compute the embedding (you need to implement get_embedding)
                        embedding_vector = get_embedding(name_text)
                        # Update the node with the computed embedding
                        query_update = f"""
                        MATCH (n:{label}) WHERE n.id = $nodeId
                        SET n.name_embedding = $embedding
                        """
                        session.run(query_update, nodeId=node_id, embedding=embedding_vector)
                        logging.info(f"Name embeddings updated for all nodes by label '{label}'.")
                    except Exception as e:
                        logging.error(f"Error creating Name embeddings for label '{label}', AND node_id {node_id}: {e}")

        logging.info("Name embeddings updated for all nodes by label.")

    def create_vector_indexes_for_all_labels(self, embedding_property="name_embedding", dimensions=1536):
        """
        Retrieve all labels from the database and create a vector index on each label using the specified embedding property.
        """
        with self.driver.session() as session:
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record["label"] for record in labels_result]
            logging.info(f"Found labels for indexing: {labels}")
            for label in labels:
                try:
                    index_name = f"text_embeddings"
                    create_vector_index(
                        self.driver,
                        index_name,
                        label=label,
                        embedding_property=embedding_property,
                        dimensions=dimensions,
                        similarity_fn="cosine"
                    )
                    logging.info(f"Vector index 'text_embeddings' created for label '{label}'.")
                except Exception as e:
                    logging.error(f"Error creating vector index for label '{label}': {e}")


# Example Usage:
if __name__ == "__main__":
    neo4j_db = Neo4jGraph()
    # Load your CSV data into Neo4j
    # neo4j_db.load_csv_to_graph("datasets/phenotype_to_genes.csv", "datasets/phenotype.csv")

    # Update nodes with embeddings computed from the 'name' property
    # neo4j_db.update_name_embeddings_by_label()

    # Create a vector index on the 'name_embedding' property for semantic search
    neo4j_db.create_vector_indexes_for_all_labels(
        embedding_property="name_embedding",
        dimensions=1536
    )
    neo4j_db.close()
