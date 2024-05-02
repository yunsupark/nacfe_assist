#adapted from original code by Krish Naik

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
#from langchain_community.vectorstores import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.chains.question_answering import load_qa_chain
#from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
#from langchain.memory import ConversationBufferMemory

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = os.getenv("PINECONE_INDEX")
index = pc.Index(pinecone_index)
                               
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    #chunks = text_splitter.split_text(text)
    pages = text_splitter.split_text(text)
    chunks = text_splitter.create_documents(pages)
    return chunks


def get_vector_store(text_chunks):

  # Create or connect to an index named "nacfe"

  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  
  # Load the document into our database
  #PineconeVectorStore.from_documents(index_name=index, documents=text_chunks, embedding=embeddings)
  vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)
  vectorstore.add_documents(text_chunks)


def main():
    st.set_page_config("Mikey")
    st.header("Upload reports for Mikey")

    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")


if __name__ == "__main__":
    main()