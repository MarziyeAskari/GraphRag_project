import os
import pandas as pd
from neo4j import GraphDatabase


class Neo4jGraph:
    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection with credentials from environment variables."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://staging.shamim.review:7474")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "lmis_neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def load_csv_to_graph(self, phenotype_to_genes_csv, phenotype_csv):
        """Load nodes and relationships from local CSV files into Neo4j."""

        # Load phenotype_to_genes.csv
        phenotype_to_genes = pd.read_csv(phenotype_to_genes_csv)
        with self.driver.session() as session:
            for _, row in phenotype_to_genes.iterrows():
                query = """
                MERGE (s:Symptom {hpo_id: $hpo_id})
                ON CREATE SET s.name = $hpo_name
                MERGE (g:Gene {gene_id: $gene_id})
                ON CREATE SET g.name = $gene_name
                MERGE (d:Disease {disease_id: $disease_id})
                MERGE (s)-[:HAS_GENE]->(g)
                MERGE (s)-[:ASSOCIATED_WITH]->(d)
                MERGE (d)-[:HAS_SYMPTOM]->(s)
                MERGE (g)-[:ASSOCIATED_WITH]->(d)
                MERGE (d)-[:HAS_GENE]->(g)
                MERGE (g)-[:IMPACTS]->(s);
                """
                session.run(query,
                            hpo_id=row["hpo_id"], hpo_name=row["hpo_name"],
                            gene_id=row["ncbi_gene_id"], gene_name=row["gene_symbol"],
                            disease_id=row["disease_id"])

        # Load phenotype.csv
        phenotype = pd.read_csv(phenotype_csv)
        with self.driver.session() as session:
            for _, row in phenotype.iterrows():
                query = """
                MERGE (d:Disease {disease_id: $database_id})
                ON CREATE SET d.name = $disease_name
                MERGE (s:Symptom {hpo_id: $hpo_id})
                MERGE (d)-[:HAS_SYMPTOM]->(s);
                """
                session.run(query,
                            database_id=row["database_id"], disease_name=row["disease_name"],
                            hpo_id=row["hpo_id"])

        print("Data successfully loaded into Neo4j.")


# Example Usage
if __name__ == "__main__":
    neo4j_db = Neo4jGraph()
    neo4j_db.load_csv_to_graph("datasets/phenotype_to_genes.csv", "datasets/phenotype.csv")
    neo4j_db.close()

