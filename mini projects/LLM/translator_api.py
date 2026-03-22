import os

from dotenv import load_dotenv
from fastapi import FastAPI

from langchain_classic.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langserve import add_routes

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
model = ChatOpenRouter(model="stepfun/step-3.5-flash:free")

# Create Prompt Template
system_template = "Translate the following into {language}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

# Create Chain
chain = prompt | model | parser

# App definition
app = FastAPI(
    title="Translation API",
    description="An API to translate text into different languages using a LLM.",
    version="1.0.0"
)

# Add routes
add_routes(
    app, 
    chain, 
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)