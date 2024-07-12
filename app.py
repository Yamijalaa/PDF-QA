import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import process_pdf, delete_collection, extract_text_from_pdf
from qa_chain import get_qa_chain
from summarizer import summarize_document
import chromadb

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def main():
    st.title("PDF Question Answering and Summarization System")

    # Session state to store the current collection name and PDF text
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'current_pdf_text' not in st.session_state:
        st.session_state.current_pdf_text = None

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                collection_name = process_pdf(uploaded_file, chroma_client)
                st.session_state.current_collection = collection_name
                st.session_state.current_pdf_text = extract_text_from_pdf(uploaded_file)
            st.success("PDF processed successfully!")

    # Document summarization
    if st.session_state.current_pdf_text:
        if st.button("Summarize Document"):
            with st.spinner("Generating summary..."):
                summary = summarize_document(st.session_state.current_pdf_text)
            st.subheader("Document Summary")
            st.write(summary)

    # Question answering
    if st.session_state.current_collection:
        st.subheader("Ask a question")
        question = st.text_input("Enter your question:")
        if question:
            with st.spinner("Thinking..."):
                qa_chain = get_qa_chain(st.session_state.current_collection, chroma_client)
                result = qa_chain({"query": question})
            st.write("Answer:", result['result'])
            st.subheader("Sources:")
            for doc in result['source_documents']:
                st.write(doc.page_content[:200] + "...")  # Display first 200 characters of each source

        # Delete current PDF
        if st.button("Delete Current PDF"):
            if delete_collection(st.session_state.current_collection, chroma_client):
                st.success("PDF deleted successfully")
                st.session_state.current_collection = None
                st.session_state.current_pdf_text = None
            else:
                st.error("Failed to delete PDF")
    else:
        st.info("Please upload and process a PDF file to start asking questions or generate a summary.")

if __name__ == "__main__":
    main()