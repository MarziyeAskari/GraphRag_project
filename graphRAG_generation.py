import re

from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from neo4j_env import graph
from get_API_key import get_API_key
import textwrap
import os
from embedding_function import get_embedding

os.environ["OPENAI_API_KEY"] = get_API_key()
entity_extraction_prompt = """
Your task is to identify the main medical entity from a user's question.
Focus on extracting ONLY the specific entity name that needs vector search.

Rules:
1. Return ONLY the entity name, nothing else
2. Prioritize disease names, then symptoms, then genes
3. Preserve exact spelling and casing
4. Ignore generic terms like "genes" or "diseases"

Examples:
Question: What genes are related to Pseudohyperkalemia?
Entity: Pseudohyperkalemia

Question: What diseases are linked to "Preauricular"?
Entity: Preauricular

Question: {question}
Entity: """

retrieval_qa_chat_prompt = """
Task: Generate a Cypher statement to query a graph database.
Instructions:
- Use vector search to find the closest matching entity if an exact match is not found.
- Use the index "name_embeddings" for similarity search on node names.

Schema:
{schema}

If the question mentions 'gene' it refers to a Gene node, 
'symptom' refers to a Symptom node, and 
'disease' refers to a Disease node.

Use this pattern for fuzzy search:
CALL db.index.vector.queryNodes("name_embeddings", 1, $embedding) 
YIELD node, score 
WHERE score > 0.7 
RETURN node.name

Example 1: 
**User Question:** What genes are related to Pseudohyperkalemia?
**Generated Query:**
```
MATCH (d:Disease)-[:HAS_GENE]->(g:Gene)
WHERE d.name = (
  CALL db.index.vector.queryNodes("name_embeddings", 1,  $embedding) 
  YIELD node, score 
  WHERE score > 0.7 
  RETURN node.name
)
RETURN g.name
```

Example 2: 
**User Question:** What diseases are linked to "Preauricular"?
**Generated Query:**
```
MATCH (s:Symptom)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE s.name = (
  CALL db.index.vector.queryNodes("name_embeddings", 1, $embedding) 
  YIELD node, score 
  WHERE score > 0.7 
  RETURN node.name
)
RETURN d.name
```

The question is:
{question}
"""


class GraphRAG:
    def __init__(self):
        # Initialize LLM for entity extraction
        self.llm = ChatOpenAI(temperature=0)

        # Entity extraction chain
        self.entity_chain = (
                PromptTemplate.from_template(entity_extraction_prompt)
                | self.llm
                | StrOutputParser()
        )
        # Cypher generation chain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=graph,
            verbose=True,
            cypher_prompt=PromptTemplate(
                input_variables=["schema", "question"],
                template=retrieval_qa_chat_prompt
            ),
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    async def extract_entity(self, question: str) -> str:
        """Use LLM to identify the exact search entity"""
        entity = await self.entity_chain.ainvoke({"question": question})
        return entity.strip('"').strip()

    async def generate_cypher_query(self, question: str) -> str:
        # Extract entity using LLM
        entity = await self.extract_entity(question)
        if not entity or entity.lower() == "none":
            return "Error: No valid entity extracted."

        # Get embedding
        embedding = get_embedding(entity)

        # Generate Cypher with parameters
        try:
            response = self.cypher_chain.invoke({
                "question": question,
                "parameters": {"embedding": embedding}
            })
            query = response.get("result", "")
            return self._validate_query(query)
        except Exception as e:
            return f"Error: {str(e)}"

    def _validate_query(self, query: str) -> str:
        """Ensure query doesn't contain Python functions"""
        if "get_embedding" in query:
            return "Invalid query: Python function in Cypher"
        return textwrap.fill(query, 60)
