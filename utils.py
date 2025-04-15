# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# #from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Google API key not found. Please check your .env file.")

# genai.configure(api_key=GOOGLE_API_KEY)

# # Extract text from PDF files
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Split text into chunks for embedding
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Generate and store vector store (FAISS index)
# def get_vector_store(text_chunks):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#         st.success("Vector store (FAISS index) saved successfully.")
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         raise e

# # Define conversational chain using Google Gemini
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "Answer is not available in the context".
    
#     Context: {context}
#     Question: {question}
#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# # Process user input (query the vector store and get a response)
# def user_input(user_question):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         #new_db = FAISS.load_local("faiss_index", embeddings)
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


#         # Perform similarity search
#         docs = new_db.similarity_search(user_question)

#         chain = get_conversational_chain()

#         # Generate response
#         response = chain(
#             {"input_documents": docs, "question": user_question},
#             return_only_outputs=True
#         )

#         st.write("Reply: ", response["output_text"])
#     except Exception as e:
#         st.error(f"Error processing user question: {e}")
#         raise e












import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st

def initialize_google_api():
    """Initializes the Google API with the provided API key."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found. Please check your .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    """Extracts text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates and stores vector store (FAISS index)."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store (FAISS index) saved successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise e

def get_conversational_chain():
    """Defines conversational chain using Google Gemini."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "Answer is not available in the context".

    Context: {context}
    Question: {question}
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """Processes user input: retrieves relevant docs and queries the conversational chain."""
    try:
        # Load the vector store using the proper embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Retrieve documents relevant to the user's question
        docs = new_db.similarity_search(user_question)

        # Get the conversational chain (make sure it's set up to accept both keys)
        chain = get_conversational_chain()

        # Call the chain with both "input_documents" and "question" keys
        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing user question: {e}")
        raise e
