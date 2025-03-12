import re
import os
import textwrap
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from neo4j_env import graph
from get_API_key import get_API_key
from embedding_function import get_embedding

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = get_API_key()

# Entity Extraction Prompt
entity_extraction_prompt = PromptTemplate.from_template(
    """
      Your task is to identify the main medical entity from a user's question and also determine its label (Gene, Symptom, or Disease).
      Focus on extracting ONLY the specific entity name and its corresponding label for vector search.
    
      Rules:
      1. Return the entity name followed by its label (e.g., "Pseudohyperkalemia: Disease").
      2. Prioritize disease names first, then symptoms, and then genes. 
         However, if the question specifically refers to a symptom (e.g., "What is related to a symptom?"), give priority to symptoms.
      3. Preserve exact spelling and casing.
      4. Ignore generic terms like "genes" or "diseases".
      5. If no valid entity is found, return "None: None".
    
      Examples:
      Question: What genes are related to Pseudohyperkalemia?
      Entity: Pseudohyperkalemia: Disease
    
      Question: What diseases are linked to "Preauricular"?
      Entity: Preauricular: Symptom
    
      Question: What symptoms are related to Dilatated internal auditory canal?
      Entity: Dilatated internal auditory canal: Symptom
    
      Question: {question}
      Entity: """
)

# Cypher Query Prompt Template
# Cypher Query Prompt Template
retrieval_qa_chat_prompt = PromptTemplate.from_template("""
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
    
    Based on the extracted entity, {entity_name} (label: {label}), generate a Cypher query that retrieves relevant data related to {entity_name}.
    Make sure to use the appropriate label (Gene, Symptom, or Disease) and relationships, depending on the label.
    
    - If the label is "Gene", the entity is a Gene node.
    - If the label is "Symptom", the entity is a Symptom node.
    - If the label is "Disease", the entity is a Disease node.

    Use appropriate relationships:
    - 'has_gene' for linking symptoms and disease to genes.
    - 'associated_with' for linking symptoms and gens to diseases.
    - 'impacts' for linking genes to symptom.
    - 'has_symptom' for linking  diseases to symptom.

    Note: Do not include any explanations or apologies in your responses.
    Do not include any text except the generated Cypher statement. Remember to correct the typo in names.

    Example 1: What genes are related to a specific symptom?
    MATCH (s:Symptom)-[:HAS_GENE]->(g:Gene)
    WHERE s.name= "Abnormal circulating lipid concentration"
    RETURN collect(g.name) AS genes

    Example 2: What diseases are associated with a specific symptom?
    MATCH (s:Symptom)-[:ASSOCIATED_WITH]->(d:Disease)
    WHERE s.name = "Preauricular pit"
    RETURN collect(d.name) AS diseases

    Example 3: What symptoms are linked to a specific gene?
    MATCH (s:Symptom)-[:HAS_GENE]->(g:Gene)
    WHERE g.name = "BRCA1"
    RETURN collect(s.name) AS symptoms

    The question is:
    {question}

    Based on the extracted entity, {entity_name}, generate a Cypher query that retrieves relevant data related to {entity_name}. Make sure to use the appropriate labels and relationships.
    """)


class GraphRAG:
    def __init__(self):
        # Initialize LLM for entity extraction
        self.llm = ChatOpenAI(temperature=0)
        self.schema = self._get_graph_schema()

        # Initialize Cypher query chain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=graph,
            verbose=True,
            cypher_prompt=retrieval_qa_chat_prompt,
            # return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    def _get_graph_schema(self) -> str:
        """Retrieve and format the Neo4j schema."""
        query = """
        CALL db.schema.visualization() 
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        result = graph.query(query)
        return str(result)

    def get_result_test(self, embedding: list, index_name: str) -> str:
        """Retrieve and format the top 3 most similar diseases from Neo4j."""

        query = """
        CALL db.index.vector.queryNodes(
            $index_name, 
            3, 
            $embedding
        )
        YIELD node, score
        RETURN node.name AS entity_name, score
        """

        try:
            results = graph.query(query, {"embedding": embedding, "index_name": index_name})

            if not results:
                return "No similar diseases found in the knowledge graph."

            return results

        except Exception as e:
            return f"Error: {str(e)}"

    def extract_entity(self, question: str) -> str:
        """Use LLM to identify the exact search entity."""
        entity_response = self.llm.invoke(entity_extraction_prompt.format(question=question))
        entity = entity_response.content.strip()
        return entity

    def extract_entity_and_label(self, entity_response: str):
        """Extract entity name and label from LLM response."""
        entity_label = entity_response.strip().split(":")
        if len(entity_label) == 2:
            return entity_label[0].strip(), entity_label[1].strip()
        return None, None

    def generate_cypher_query(self, question: str) -> str:
        """Generate Cypher query using extracted entity and its vector embedding."""
        entity_output = self.extract_entity(question)
        entity, label = self.extract_entity_and_label(entity_output)
        if not entity or entity.lower() == "none":
            return "Error: No valid entity extracted."

        # Get embedding
        # Determine the appropriate index name
        index_mapping = {
            "Gene": "text_embeddings_gene",
            "Symptom": "text_embeddings_symptom",
            "Disease": "text_embeddings_disease"
        }
        index_name = index_mapping.get(label, None)
        embedding = get_embedding(entity)
        results = self.get_result_test(embedding, index_name)
        entity_name = results[0]['entity_name']

        if not index_name:
            return "Error: Unsupported entity type."

        # Generate Cypher with parameters
        try:
            response = self.cypher_chain.invoke({
                "schema": self.schema,
                "question": question,
                "label": label,
                "query": question,
                "entity_name": entity_name
            })
            print("Full LangChain Response:", response)
            query = response.get("result", "")
            return self._validate_query(query)
        except Exception as e:
            return f"Error: {str(e)}"

    def _validate_query(self, query: str) -> str:
        """Ensure query doesn't contain Python functions."""
        if "get_embedding" in query:
            return "Invalid query: Python function in Cypher"
        return textwrap.fill(query, 60)
