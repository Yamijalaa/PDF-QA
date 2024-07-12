from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def summarize_document(text):
    # Initialize the language model
    llm = OpenAI(temperature=0)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(text)

    # Create Document objects
    docs = [Document(page_content=t) for t in texts]

    # Initialize the summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Generate the summary
    summary = chain.run(docs)

    return summary