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
If the question refers to 'has_gene', 'associated_with', 'has_symptom', or 'impacts', use them as relationships.

Note: Do not include any explanations or apologies in your responses.
Do not include any text except the generated Cypher statement. Remember to correct the typo in names.

Example 1: What genes are related to a specific symptom?
MATCH (s:Symptom)-[:HAS_GENE]->(g:Gene)
WHERE s.symptom_name = "Abnormal circulating lipid concentration"
RETURN g.gene_id, g.gene_name

Example 2: What diseases are associated with a specific symptom?
MATCH (s:Symptom)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE s.symptom_name = "Preauricular pit"
RETURN d.disease_id, d.disease_name

Example 3: What symptoms are linked to a specific gene?
MATCH (g:Gene)-[:HAS_GENE]->(s:Symptom)
WHERE g.gene_name = "BRCA1"
RETURN s.symptom_name

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
        response = self.cypher_chain.run(question)
        return textwrap.fill(response, 60)
