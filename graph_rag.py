import os
import logging
from get_API_key import get_API_key  # Ensure this function retrieves your OpenAI API key securely
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_database import Neo4jGraph

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = get_API_key()

driver = Neo4jGraph().driver

# Initialize the embedder using OpenAI embeddings
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Define your query
query_text = "What are the common symptoms and associated genes for Pseudohyperkalemia familial?"

# Define the Cypher retrieval query
retrieval_query = """
MATCH (d:Disease)
WHERE d.name CONTAINS $query_text
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)<-[:ASSOCIATED_WITH]-(g:Gene)
RETURN d.name AS disease, collect(DISTINCT s.name) AS symptoms, collect(DISTINCT g.name) AS genes
"""

# Initialize the Vector Cypher Retriever with the retrieval query
vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="text_embeddings",  # Ensure this index exists in your Neo4j database
    embedder=embedder,
    retrieval_query=retrieval_query
)

# Initialize the LLM
llm = OpenAILLM(model_name="gpt-4", model_params={"temperature": 0.0})

# Define the prompt template
prompt_template = RagTemplate(
    template="""
Using the following context, answer the query.

Context:
{context}

Query:
{query_text}

Answer:
""",
    expected_inputs=["context", "query_text"]
)

# Initialize the GraphRAG pipeline
graph_rag = GraphRAG(
    llm=llm,
    retriever=vector_cypher_retriever,
    prompt_template=prompt_template
)

# Execute the search
response = graph_rag.search(query_text=query_text, retriever_config={'top_k': 1})

# Output the answer
print("GraphRAG Answer:")
print(response.answer)

# Close the Neo4j driver connection
driver.close()
