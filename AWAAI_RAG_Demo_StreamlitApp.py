import openai
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import tempfile
load_dotenv()

# Initialize OpenAI API (Make sure to replace 'your-api-key' with your actual API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load the PDF
def load_pdf(pdf_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    
    # Load the PDF using the temporary file path
    pdf_loader = PyPDFLoader(temp_file_path)
    documents = pdf_loader.load()
    return documents

# Function to split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to initialize vector store
def initialize_vector_store(text_chunks):
    embedding_model = OpenAIEmbeddings(api_key=openai.api_key)
    vector_store = Chroma.from_documents(documents=text_chunks, embedding=embedding_model)
    return vector_store

# Function to generate the answer using OpenAI GPT model
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

    answer_with_context = text_response.choices[0].message.content
    return answer_with_context

# Streamlit UI
def main():
    # Title
    st.title("Document-Based Question Answering")

    # Add the subtitle
    st.subheader("AWAAI RAG Demo by Zeba Karkhanawala")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file is not None:
        # Load PDF
        documents = load_pdf(pdf_file)

        # Split documents into chunks
        text_chunks = split_documents(documents)

        # Initialize vector store
        vector_store = initialize_vector_store(text_chunks)

        # Input question
        query = st.text_input("Ask a question:")

        if query:
            # Retrieve documents relevant to the question
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(query)

            # Generate answer based on context and query
            answer = generate_answer(query, relevant_docs)

            # Display the answer
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
