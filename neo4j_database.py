from neo4j import GraphDatabase

class Neo4jGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_sample_data(self):
        """Create sample nodes and relationships in Neo4j."""
        query = """
        MERGE (alice:Person {name: 'Alice', role: 'Engineer'})
        MERGE (bob:Person {name: 'Bob', role: 'Manager'})
        MERGE (techcorp:Company {name: 'TechCorp'})
        MERGE (alice)-[:WORKS_AT]->(techcorp)
        MERGE (bob)-[:WORKS_AT]->(techcorp)
        MERGE (alice)-[:REPORTS_TO]->(bob)
        """
        with self.driver.session() as session:
            session.run(query)

    def query_graph(self, person_name):
        """Retrieve relationships for a specific person."""
        query = """
        MATCH (p:Person {name: $name})-[r]->(c) 
        RETURN type(r) AS relationship, labels(c)[0] AS node_type, c.name AS node_name
        """
        with self.driver.session() as session:
            result = session.run(query, name=person_name)
            return [{"relationship": record["relationship"], "node_type": record["node_type"], "node_name": record["node_name"]} for record in result]

# Example Usage
if __name__ == "__main__":
    neo4j_db = Neo4jGraph()
    neo4j_db.create_sample_data()
    results = neo4j_db.query_graph("Alice")
    print(results)  # Example output
    neo4j_db.close()
