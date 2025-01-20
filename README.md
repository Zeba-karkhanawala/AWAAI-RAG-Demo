# RAG Workshop for Asian Women Advancing AI - Demo by Zeba Karkhanawala

## Introduction to the Workshop

Welcome to the **RAG (Retrieval-Augmented Generation) Workshop for Asian Women Advancing AI**! This demo has been created by **Zeba Karkhanawala** to demonstrate the power of building your own Streamlit-based chatbot, leveraging **Retrieval-Augmented Generation (RAG)** techniques. The goal is to enable users to upload their documents, and interact with them using a simple yet effective AI-powered application that can answer any query related to the content of the uploaded document.

---

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an advanced AI technique that combines information retrieval (searching for relevant documents) with language generation (creating coherent, relevant answers from the retrieved documents). The RAG model improves the quality of responses from an AI system by integrating external documents or databases and using that information to answer user queries. This method can be particularly useful for tasks like question answering, summarization, and information extraction.

---

## What is This Demo About?

This demo demonstrates how you can build your very own **Streamlit Chatbot** that can query any document uploaded by the user. The chatbot will intelligently search through the uploaded document and return answers to your queries using **RAG** technology. The demo is powered by **Streamlit**, a powerful tool for building interactive web applications in Python.

---

## Powered By

- **Streamlit**: A Python framework to create interactive web apps for machine learning and data science projects.
- **OpenAI GPT-4o**: To generate responses to user queries using large language models.
- **Chroma**: A vector store used to store the document embeddings for quick retrieval.
- **Langchain**: A framework to simplify the process of combining LLMs and external tools like APIs, databases, and document loaders.

---

## Instructions to Run the Demo Locally

Follow these steps to set up and run the Streamlit app on your local machine:

### 1. Clone the Repository

To get started, clone the repository to your local machine using Git:

```bash
git clone https://github.com/yourusername/AWAAI-RAG-Demo.git
```

### 2. Set Up the .env File

The demo requires an **OpenAI API Key** to interact with the OpenAI models. To set up the API key, follow the steps mentioned in the **'How to get an OpenAI API key.pdf'** document. Then follow the below steps.

1. Create a new file named `.env` in the **root directory** of the project.
2. Add the following line to the `.env` file, replacing `your-openai-api-key` with your actual OpenAI API key:

```txt
OPENAI_API_KEY="your-openai-api-key"
```
### 3. Install the Required Libraries

This project requires certain libraries to run. You can install them using the requirements.txt file included in the repository.

a) First, navigate to the project directory:
```bash
cd AWAAI-RAG-Demo
```

b) Then, install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
Once the dependencies are installed, you can run the Streamlit app with the following command:

```bash
streamlit run AWAAI_RAG_Demo_StreamlitApp.py
```

This command will start the app and you can access it in your web browser at http://localhost:8501.