import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import uuid

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_pdf(uploaded_file, chroma_client):
    text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Generate a unique collection name
    collection_name = f"pdf_qa_{uuid.uuid4().hex}"

    # Create and persist a Chroma collection
    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name=collection_name
    )

    return collection_name

def delete_collection(collection_name, chroma_client):
    try:
        chroma_client.delete_collection(collection_name)
        return True
    except ValueError:
        return False