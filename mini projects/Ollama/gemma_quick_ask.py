""""
Gemma QuickAsk

A lightweight Streamlit Q&A app that uses LangChain + Ollama (gemma:2b) to answer user-entered questions.  
The app:

- Loads environment variables (including optional LangSmith tracing settings)
- Defines a simple chat prompt (system + user question)
- Builds a LangChain pipeline: Prompt → OllamaLLM → String Output Parser
- Takes input from a Streamlit text box and displays the model's response

In short, it is a minimal local-LLM question-answer mini project with a clean web UI.
"""

import os
import streamlit as st

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY") or ""
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING") or ""
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT") or ""

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a help assisstant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo with OLLAMA Gemma:2b model")
input_text = st.text_input("What question you've in mind?")

# Ollama LLAMA2 model
llm = OllamaLLM(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))