from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from neo4j_env import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import textwrap


class VectorRAG:
    def __init__(self):
        # Initialize the Neo4jVector store by loading existing graph data
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(),  # Use OpenAI embeddings for vectorizing the text data
            url=NEO4J_URI,  # Neo4j database URI
            username=NEO4J_USERNAME,  # Username for Neo4j authentication
            password=NEO4J_PASSWORD,  # Password for Neo4j authentication
            index_name="name_embeddings",  # Name of the vector index in Neo4j
            node_label=["Symptom", "Disease", "Gene"],  # Multiple node labels in the Neo4j graph
            text_node_properties=["name"],  # Text properties of the nodes to be embedded
            embedding_node_property="name_embedding",  # Property that stores the embeddings in Neo4j
        )

        # Load the retrieval-qa-chat prompt from Langchain hub
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Create a document combination chain for summarizing retrieved documents
        self.combine_docs_chain = create_stuff_documents_chain(ChatOpenAI(temperature=0), self.retrieval_qa_chat_prompt)

        # Set up the retrieval chain with the vector store and document combination chain
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(),  # Use the vector store as a retriever for similarity search
            combine_docs_chain=self.combine_docs_chain  # Combine retrieved documents using the QA chain
        )

    def query(self, question: str) -> str:
        # Perform the query using the retrieval chain
        result = self.retrieval_chain.invoke(input={"input": question})

        # Format the answer to make it more readable
        return textwrap.fill(result['answer'], 60)
