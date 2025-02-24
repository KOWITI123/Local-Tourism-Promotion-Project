from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv  # Load environment variables

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ ERROR: Missing OpenAI API Key. Ensure it is set in the .env file.")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Replace with actual connection string
db = client['kenya_tourism']
collection = db['tourist_sites']

# Retrieve all tourist sites data
documents = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Convert MongoDB documents to LangChain format
docs = [Document(page_content=json.dumps(doc), metadata={"source": "kenya_tourism"}) for doc in documents]

# Store in ChromaDB
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Create RAG Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Function to Query RAG Pipeline
def ask_question(query):
    return qa.invoke({"query": query})["result"]


# Interactive Terminal Loop
print("\n💬 Chat with your tourism database! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Exiting chat. Have a great day!")
        break

    response = ask_question(query)
    print(f"AI: {response}\n")

