import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from gtts import gTTS
import threading
import tempfile
import pygame

st.set_page_config(
    page_title="OpenAI_ChatBOT",
    page_icon="👽",
    layout="centered",
)

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatOpenAI(api_key=api_key, model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Initialize chat session state if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []

st.markdown(
    """
    <style>
    .container {
        margin: 0;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid pink;
        position: relative;
        overflow: hidden;
    }

    .container h4,
    .container p {
        position: relative;
        z-index: 1;
        color: #fff;
        transition: color 0.5s ease;
    }
    </style>
    
    <div class="container">
        <h4>OpenAI ChatBOT</h4>
        <p>Lets deep dive into the world of AI 👽</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings
st.sidebar.title("Settings ")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="api_key")
engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-3.5-turbo"], key="engine")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, key="temperature")
max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=200, value=120, key="max_tokens")

# Display chat history
if st.session_state.chat_session:
    for message in st.session_state.chat_session:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

# User input
user_prompt = st.chat_input('Message OpenAI ChatBot...')
if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    if api_key:
        with st.spinner('Generating response...'):
            response_text = generate_response(user_prompt, api_key, engine, temperature, max_tokens)
            st.session_state.chat_session.append({"role": "user", "text": user_prompt})
            st.session_state.chat_session.append({"role": "assistant", "text": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.last_response = response_text
    else:
        st.warning("Please enter the OpenAI API Key in the sidebar")
