import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set API keys
# os.environ["GROQ_API_KEY"] = "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"

st.title("Document QA with LangChain")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and split the document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_file_path)
    else:
        loader = TextLoader(temp_file_path)

    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)

    # Create embeddings and vector store
    # embeddings = OpenAIEmbeddings(openai_api_key=apikey)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = FAISS.from_documents(documents, embeddings)

    # Set up the LLM
    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context do not involve a single word by your own same to same word should be output:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Input question from user
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        response = retrieval_chain.invoke({"input": question})
        st.write(response["answer"])
