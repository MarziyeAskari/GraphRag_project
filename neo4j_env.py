from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from get_API_key import get_API_key

load_dotenv('.env', override=True)
# Warning control
import warnings

warnings.filterwarnings("ignore")

NEO4J_URI = 'bolt://staging.shamim.review:7687'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'lmis_neo4j'
NEO4J_DATABASE = os.getenv('neo4j')
OPENAI_API_KEY = get_API_key()
# OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

# Global constants
VECTOR_INDEX_NAME = 'text_embeddings'
VECTOR_NODE_LABEL = 'name_embeddings'
VECTOR_SOURCE_PROPERTY = 'name_embeddings'
VECTOR_EMBEDDING_PROPERTY = 'name_embeddings'

graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
