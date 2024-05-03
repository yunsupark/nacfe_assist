#adapted from original code by Krish Naik

import streamlit as st
#from PyPDF2 import PdfReader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
#from langchain_community.vectorstores import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate#, ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pinecone_index = os.getenv("PINECONE_INDEX")
index = pc.Index(pinecone_index)
                               

def get_conversational_chain(user_question=None):
    prompt_template = """
    Answer the question in 50 words or less from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I can't find the answer in our reports. Please consider rewording your question and try again", don't provide the wrong answer. 
    \n
    Return a "SOURCES" part in your answer on a new line if you are able to answer the question. Otherwise SOURCES is not required.
    The "SOURCES"  should be the title of the documents used in the response.
    \n
    CHAT HISTORY:\n {chat_history}\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Access memory from session state
    memory = st.session_state.get("memory", ConversationBufferMemory(memory_key="chat_history", input_key="question"))


    # Use the retrieved chat history
    #chat_history = memory if memory else []

    prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=prompt_template)

    # Chain the conversation history
    chain = load_qa_chain(model, chain_type="stuff", memory=memory, prompt=prompt)

    # Update memory with new conversation (optional)
    #memory.chat_memory.add_user_message("question")
    #memory.chat_memory.add_ai_message("output_text")

    # Persist memory in session state
    st.session_state["memory"] = memory

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vstore = PineconeVectorStore.from_existing_index(pinecone_index, embeddings)

    docs = vstore.similarity_search(user_question, 3)
    chain = get_conversational_chain(user_question)

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    
    #show reply
    st.write("Reply: ", response["output_text"])
    return response

def main():
    st.set_page_config("Mikey")
    st.header("Chat with Mikey, the NACFE AI Assistant")
    st.subheader("Mikey can answer questions using information found in the reports shown on the left")

    # Chat widget for user input
    user_question = st.chat_input(placeholder="What would you like to know?")

    if user_question:

        # Get response from conversation chain
        user_input(user_question)    
        
        # Print response for debugging (optional)
        #print(response)

    memory = st.session_state.get("memory", ConversationBufferMemory(memory_key="chat_history", input_key="question"))

    conversation_history = memory
    print(conversation_history)

    # Display conversation history
    #for chat_memory in conversation_history:
        #if message == "HumanMessage":
            #st.write("Question",["content"])
        #else:
            #st.write("Reply", ["content"])
    with st.sidebar:
            st.image('images/NACFE logo_newtagline.png')
            st.title("Available Reports")
            st.write("RoL-E MD Box Truck Report")
            st.write("RoL-E HD Regional Haul Report")
            st.write("RoL-E Vans & Step Vans Report")
            st.write("RoL-E Terminal Tractor Report")
            st.write("Run on Less - Electric Summary")

if __name__ == "__main__":
    main()