from neo4j_database import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize Neo4j Connection
graph_db = Neo4jGraph()

def retrieve_knowledge(entity_name):
    """Query Neo4j and format results as knowledge text."""
    results = graph_db.query_graph(entity_name)
    if not results:
        return "No information found."
    knowledge = "\n".join([f"{entity_name} {r['relationship']} {r['node_name']} ({r['node_type']})" for r in results])
    return knowledge

def generate_response(user_query, entity_name):
    """Retrieve knowledge from Neo4j and generate a response using LLM."""
    knowledge = retrieve_knowledge(entity_name)

    prompt = PromptTemplate(
        input_variables=["query", "knowledge"],
        template="Using the following knowledge:\n{knowledge}\nAnswer the query: {query}"
    )

    llm = ChatOpenAI(model_name="gpt-4")
    return llm.predict(prompt.format(query=user_query, knowledge=knowledge))

# Example Usage
user_query = "Where does Alice work?"
response = generate_response(user_query, "Alice")
print(response)

# Close DB connection
graph_db.close()
