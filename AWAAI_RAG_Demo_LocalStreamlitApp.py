__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import openai
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import tempfile
import chromadb
from chromadb.config import Settings

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif ext == ".xlsx":
        loader = UnstructuredExcelLoader(temp_file_path)
    elif ext == ".csv":
        loader = CSVLoader(temp_file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(temp_file_path)
    else:
        st.error("Unsupported file format. Please upload a PDF, Excel, CSV, or DOCX file.")
        return None
    
    return loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def initialize_vector_store(text_chunks):
    embedding_model = OpenAIEmbeddings(api_key=openai.api_key)
    return Chroma.from_documents(documents=text_chunks, embedding=embedding_model)

def generate_answer(query, relevant_docs):
    contextual_query = f"Context: {relevant_docs}\n\nQuery: {query}"
    prompt_with_context = [
        {"role": "system", "content": "You are an expert assistant that answers the query based on the context provided."},
        {"role": "user", "content": contextual_query}
    ]
    
    text_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=prompt_with_context,
    )
    
    return text_response.choices[0].message.content

def main():
    st.title("Multi-Format Document-Based Question Answering")
    st.subheader("AWAAI RAG Demo by Zeba Karkhanawala")
    
    file = st.file_uploader("Upload a document", type=["pdf", "xlsx", "csv", "docx"])
    
    if file is not None:
        documents = load_document(file)
        if documents:
            text_chunks = split_documents(documents)
            vector_store = initialize_vector_store(text_chunks)
            query = st.text_input("Ask a question:")
            
            if query:
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(query)
                answer = generate_answer(query, relevant_docs)
                st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
