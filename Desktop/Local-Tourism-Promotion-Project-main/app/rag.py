import psycopg2
import json
import os
from dotenv import load_dotenv  # Load environment variables

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app.db_config import DB_CONFIG  # Import DB_CONFIG from your Flask app

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ ERROR: Missing OpenAI API Key. Ensure it is set in the .env file.")

# Connect to PostgreSQL
def create_connection():
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise

# Retrieve all tourist sites data from PostgreSQL
def fetch_attractions():
    connection = create_connection()
    try:
        cursor = connection.cursor()
        query = """
        SELECT name, location, description, activities, best_time_to_visit, 
               rates_citizens, rates_residents, rates_non_residents
        FROM attractions
        """
        cursor.execute(query)
        results = cursor.fetchall()
        # Convert results to a list of dictionaries
        documents = [
            {
                "name": row[0],
                "location": row[1],
                "description": row[2],
                "activities": row[3],
                "best_time_to_visit": row[4],
                "rates_citizens": row[5],
                "rates_residents": row[6],
                "rates_non_residents": row[7]
            }
            for row in results
        ]
        return documents
    except psycopg2.Error as e:
        print(f"Error fetching data: {e}")
        raise
    finally:
        cursor.close()
        connection.close()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Fetch data and convert to LangChain Document format
documents = fetch_attractions()
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

# Optional: Interactive Terminal Loop for testing
if __name__ == "__main__":
    print("\n💬 Chat with your tourism database! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Exiting chat. Have a great day!")
            break
        response = ask_question(query)
        print(f"AI: {response}\n")