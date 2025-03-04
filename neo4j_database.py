import os
import pandas as pd
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
from neo4j_graphrag.indexes import create_vector_index
from embedding_function import get_embedding  # Ensure this function is defined properly

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

    def load_csv_and_embed(self, phenotype_to_genes_csv, phenotype_csv):
        """Load nodes and relationships into Neo4j while computing and storing embeddings."""
        try:
            phenotype_to_genes = pd.read_csv(phenotype_to_genes_csv, encoding="utf-8-sig")
            phenotype = pd.read_csv(phenotype_csv, encoding="utf-8-sig")
        except Exception as e:
            logging.error(f"Error reading CSV files: {e}")
            return

        # Compute embeddings before inserting into Neo4j
        def add_embeddings(df, name_col):
            if name_col in df.columns:
                df["name_embedding"] = df[name_col].apply(lambda x: get_embedding(x) if pd.notna(x) else None)

        add_embeddings(phenotype_to_genes, "hpo_name")
        add_embeddings(phenotype_to_genes, "gene_symbol")
        phenotype_to_genes.to_csv('phenotype_to_genes_embed.csv')
        add_embeddings(phenotype, "disease_name")
        phenotype.to_csv('phenotype.csv')

        query = """
        UNWIND $rows AS row
        MERGE (s:Symptom {id: row.hpo_id})
        ON CREATE SET s.name = row.hpo_name, s.name_embedding = row.hpo_embedding
        MERGE (g:Gene {id: row.gene_id})
        ON CREATE SET g.name = row.gene_symbol, g.name_embedding = row.gene_embedding
        MERGE (d:Disease {id: row.disease_id})
        ON CREATE SET d.name = row.disease_name, d.name_embedding = row.disease_embedding
        MERGE (s)-[:HAS_GENE]->(g)
        MERGE (s)-[:ASSOCIATED_WITH]->(d)
        MERGE (d)-[:HAS_SYMPTOM]->(s)
        MERGE (g)-[:ASSOCIATED_WITH]->(d)
        MERGE (d)-[:HAS_GENE]->(g)
        MERGE (g)-[:IMPACTS]->(s);
        """

        with self.driver.session() as session:
            try:
                batch_size = 1000  # Process data in batches
                for batch in range(0, len(phenotype_to_genes), batch_size):
                    batch_data = phenotype_to_genes.iloc[batch: batch + batch_size]
                    session.run(
                        query,
                        rows=batch_data.rename(
                            columns={
                                "hpo_name": "hpo_embedding",
                                "gene_symbol": "gene_embedding",
                                "disease_name": "disease_embedding",
                            }
                        ).to_dict("records"),
                    )

                for batch in range(0, len(phenotype), batch_size):
                    batch_data = phenotype.iloc[batch: batch + batch_size]
                    session.run(
                        query,
                        rows=batch_data.rename(
                            columns={"disease_name": "disease_embedding"}
                        ).to_dict("records"),
                    )

                logging.info("Data and embeddings successfully loaded into Neo4j.")
            except Exception as e:
                logging.error(f"Error executing Neo4j queries: {e}")

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
                    index_name = f"text_embeddings_{label.lower()}"
                    create_vector_index(
                        self.driver,
                        index_name,
                        label=label,
                        embedding_property=embedding_property,
                        dimensions=dimensions,
                        similarity_fn="cosine"
                    )
                    logging.info(f"Vector index '{index_name}' created for label '{label}'.")
                except Exception as e:
                    logging.error(f"Error creating vector index for label '{label}': {e}")


# Example Usage:
if __name__ == "__main__":
    neo4j_db = Neo4jGraph()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_directory, "datasets")
    # phenotype = pd.read_csv(f"{output_dir}\\phenotype.csv")

    # Load CSV and embed while inserting
    neo4j_db.load_csv_and_embed(f"{output_dir}\\phenotype_to_genes.csv", f"{output_dir}\\phenotype.csv")

    # Create a vector index on the 'name_embedding' property
    neo4j_db.create_vector_indexes_for_all_labels(
        embedding_property="name_embedding",
        dimensions=1536
    )

    neo4j_db.close()
