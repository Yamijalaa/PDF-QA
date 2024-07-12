from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def get_qa_chain(collection_name, chroma_client):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Load the persisted Chroma collection
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever()

    # Initialize the language model
    llm = OpenAI(temperature=0)

    # Create and return the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain