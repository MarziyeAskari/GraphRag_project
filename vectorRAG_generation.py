class VectorRAG:
    def __init__(self):
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(),  # Use OpenAI embeddings for vectorization
            url=NEO4J_URI,  # URI for the Neo4j instance
            username=NEO4J_USERNAME,  # Username for authenticating with Neo4j
            password=NEO4J_PASSWORD,  # Password for authenticating with Neo4j
            index_name=VECTOR_INDEX_NAME,  # The name of the vector index in Neo4j
            node_label=VECTOR_NODE_LABEL,  # The label used for the nodes in the graph
            text_node_properties=[VECTOR_SOURCE_PROPERTY],  # Text properties of the nodes that will be embedded
            embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
            # Property that stores the embedding vectors on the nodes
        )

        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        self.combine_docs_chain = create_stuff_documents_chain(ChatOpenAI(temperature=0), self.retrieval_qa_chat_prompt)

        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(),  # Use the vector store as a retriever for similarity search
            combine_docs_chain=self.combine_docs_chain  # Combine the retrieved documents using the QA chat chain
        )

    def query(self, question: str) -> str:
        result = self.retrieval_chain.invoke(input={"input": question})
        return textwrap.fill(result['answer'], 60)
