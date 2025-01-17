# @Author: Dhananjay Kumar
# @Date: 17-01-2025
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 17-01-2025
# @Title: Python program to designing a prompt for an LLM to generate a marketing campaign idea for a new product. Describe your approach to creating this prompt, including any specific techniques you would use to ensure creativity and relevance in the generated output.


import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama

LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Load environment variables
load_dotenv()

# Langsmith Tracking Configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACKING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default_project")

# Streamlit UI Setup
st.title("Langchain Chatbot")
input_text = st.text_input("What question do you have in your mind related to marketing?")

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helping assiatnce for a marketing campaign idea for a new product. Please respond to the question asked ."),
    ("user", "Question: {question}")
])
# Configure LLM (Ollama Model)
try:
    llm = Ollama(model="gemma:2b")
except Exception as e:
    st.error("Failed to initialize Ollama model. Please check your setup.")
    st.stop()

# Combine Prompt and LLM into a Chain
chain = LLMChain(prompt=prompt, llm=llm)

# Process User Input and Generate Response
if input_text:
    try:
        response = chain.run({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
