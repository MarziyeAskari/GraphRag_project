import os
import logging
from get_API_key import get_API_key  # Ensure this function retrieves your OpenAI API key securely
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

# Set up logging


# Set your OpenAI API Key securely
os.environ["OPENAI_API_KEY"] = get_API_key()
# Initialize components
graph = Neo4jGraph(url="bolt://staging.shamim.review:7687", username="neo4j", password="lmis_neo4j")
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

result = chain.invoke({"query": "tell me all Genes is related to Pseudohyperkalemia, familial, 2, due to red cell leak"})
print(f"Intermediate steps: {result['intermediate_steps']}")
print(f"Final answer: {result['result']}")
