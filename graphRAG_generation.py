from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from neo4j_env import graph
from get_API_key import get_API_key
import textwrap
import os

os.environ["OPENAI_API_KEY"] = get_API_key()
retrieval_qa_chat_prompt = """
Task: Generate Cypher statement to 
query a graph database.
Instructions:
Use only the provided relationship types and properties in the 
schema. Do not use any other relationship types or properties that
are not provided.
Remember the relationships are like Schema:
{schema}
if the question mentions 'gene' it refers to a Gene node, 
'symptom' refers to a Symptom node, and 
'disease' refers to a Disease node.

IMPORTANT: If an entity name in the question appears to be a disease, it **must** be assigned to a Disease node (`d:Disease`), even if it could be confused with a symptom.

If the question refers to 'has_gene', 'associated_with', 'has_symptom', or 'impacts', use them as relationships.

Note: Do not include any explanations or apologies in your responses.
Do not include any text except the generated Cypher statement. Remember to correct the typo in names.

Example 1: What genes are related to a specific disease?
MATCH (d:Disease)-[:HAS_GENE]->(g:Gene)
WHERE d.name = "Pseudohyperkalemia, familial, 2, due to red cell leak"
RETURN d.id, d.name, g.id, g.name

Example 2: What diseases are associated with a specific symptom?
MATCH (s:Symptom)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE s.name = "Preauricular pit"
RETURN d.id, d.name

Example 3: What symptoms are linked to a specific gene?
MATCH (g:Gene)-[:HAS_GENE]->(s:Symptom)
WHERE g.name = "BRCA1"
RETURN s.name

The question is:
{question}

"""



class GraphRAG:
    def __init__(self):
        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=retrieval_qa_chat_prompt
        )
        self.cypher_chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0),
            graph=graph,
            verbose=True,
            cypher_prompt=self.cypher_prompt,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    def generate_cypher_query(self, question: str) -> str:
        response = self.cypher_chain.invoke(question)

        # Check if the response is a dictionary and extract the Cypher query
        if isinstance(response, dict):
            cypher_query = response.get("result", "")
        else:
            cypher_query = response  # Assume it's already a string

        if cypher_query and cypher_query != "I don't know the answer.":
            return textwrap.fill(cypher_query, 60)  # Ensure we return a string

        return "I don't know the answer."
