import os
import logging
from get_API_key import get_API_key
from neo4j_database import Neo4jGraph
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = get_API_key()

# Connect to the Neo4j database
driver = Neo4jGraph().driver

# Initialize the embedder using OpenAI embeddings
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Define available indices
indices = {
    "disease": "Disease_name_embeddings",
    "symptom": "Symptom_name_embeddings",
    "gene": "Gene_name_embeddings"
}

def determine_index(query):
    """
    Determine the appropriate index based on the query content.
    """
    query_lower = query.lower()
    if "disease" in query_lower:
        return indices["disease"]
    elif "symptom" in query_lower:
        return indices["symptom"]
    elif "gene" in query_lower:
        return indices["gene"]
    else:
        # Default to disease index if no specific keyword is found
        return indices["disease"]

# Define your query
query_text = "What are the common symptoms and associated genes for Pseudohyperkalemia, familial?"

# Determine the appropriate index based on the query
selected_index = determine_index(query_text)

# Initialize the Vector Retriever with the selected index
vector_retriever = VectorRetriever(
    driver=driver,
    index_name=selected_index,
    embedder=embedder,
    return_properties=["name", "id"]
)

# Set similarity threshold (adjust this value as needed)
similarity_threshold = 0.75

# Build the retrieval query by inserting the threshold as a literal value
retrieval_query = f"""
MATCH (d:Disease)
WHERE d.name_embedding IS NOT NULL
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)<-[:ASSOCIATED_WITH]-(g:Gene)
RETURN d.name AS disease, collect(DISTINCT s.name) AS symptoms, collect(DISTINCT g.name) AS genes
"""

# Initialize the Vector Cypher Retriever using the modified query
vector_cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name=selected_index,
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

# Initialize the GraphRAG pipelines
vector_rag = GraphRAG(
    llm=llm,
    retriever=vector_retriever,
    prompt_template=prompt_template
)

cypher_rag = GraphRAG(
    llm=llm,
    retriever=vector_cypher_retriever,
    prompt_template=prompt_template
)

# Execute the search using Vector RAG (no threshold is passed here)
vector_response = vector_rag.search(query_text=query_text, retriever_config={'top_k': 5})
print("Vector RAG Answer:")
print(vector_response.answer)

# Execute the search using Cypher RAG (threshold is already embedded in the query)
cypher_response = cypher_rag.search(query_text=query_text, retriever_config={'top_k': 1})
print("\nCypher RAG Answer:")
print(cypher_response.answer)

# Close the Neo4j connection
driver.close()
