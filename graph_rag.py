import os
import logging
from neo4j import GraphDatabase
from get_API_key import get_API_key  # Ensure this function retrieves your OpenAI API key securely
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_database import Neo4jGraph

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API Key securely
os.environ["OPENAI_API_KEY"] = get_API_key()

# Initialize components
driver = Neo4jGraph().driver
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
llm = OpenAILLM(model_name="gpt-4")

retrieval_query = """
MATCH (d:Disease)
WHERE d.name CONTAINS $query
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (d)-[:ASSOCIATED_WITH]->(g:Gene)
RETURN d.name AS disease, 
       collect(s.name) AS symptoms, 
       collect(g.name) AS genes
"""

retriever = VectorCypherRetriever(
    driver=driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

prompt_template = RagTemplate(
    template="""
    Analyze this medical query using the following context:
    Disease: {disease}
    Symptoms: {symptoms}
    Genes: {genes}

    Formulate a clinical response using ONLY this information.
    Question: {query_text}
    Answer:
    """,
    expected_inputs=["query_text", "disease", "symptoms", "genes"]
)

graph_rag = GraphRAG(
    llm=llm,
    retriever=retriever,
    prompt_template=prompt_template
)

try:


    # response = graph_rag.search(
    #     query_text="Premature centromere division",
    #     retriever_config={"query_params": {"query": "Premature centromere division"}}
    # )
    # print("Final Answer:", response.answer)
finally:
    driver.close()
