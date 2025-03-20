from langchain_openai import ChatOpenAI  # Use ChatOpenAI instead of OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import psycopg2
from typing import List, Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file (optional backup)
load_dotenv()
# Get API key from environment, stripping quotes
api_key = os.getenv('OPENAI_API_KEY', '').strip('"')
print(f"Loaded API Key: {api_key}")

# Database configuration
DB_CONFIG = {
    'dbname': 'kenya_tourism',
    'user': 'postgres',
    'password': 'remykowiti123',  # Replace with your actual PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}

# Define JSON output structure for database results
class TourismAttraction(BaseModel):
    name: str = Field(description="Name of the destination or experience.")
    location: str = Field(description="Location of the destination.")
    description: str = Field(description="Brief description of the destination or experience.")
    activities: list = Field(description="List of activities available at the destination.")
    best_time_to_visit: str = Field(description="Recommended time to visit.")
    rates: dict = Field(description="Entry or participation fees for different categories of visitors.")

# OpenAI Model Configuration (use ChatOpenAI for chat models like gpt-4)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key)

# Output parser for plain text response
output_parser = StrOutputParser()

# Function to create a database connection
def create_connection():
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

# Function to query tourism attractions from the database
def query_attractions(location: Optional[str] = None, budget: Optional[int] = None) -> List[dict]:
    attractions = []
    connection = None
    cursor = None
    
    try:
        connection = create_connection()
        if not connection:
            return attractions
        
        cursor = connection.cursor()
        
        query = """
        SELECT name, location, description, activities, best_time_to_visit, 
               rates_citizens, rates_residents, rates_non_residents
        FROM attractions
        """
        params = []
        
        # Add conditions based on location and budget
        conditions = []
        if location:
            conditions.append("location ILIKE %s")
            params.append(f"%{location}%")
        if budget:
            conditions.append("""
                (rates_citizens != 'N/A' AND CAST(REPLACE(REPLACE(REPLACE(rates_citizens, 'Ksh ', ''), ' per night', ''), ',', '') AS INTEGER) <= %s)
                OR (rates_residents != 'N/A' AND CAST(REPLACE(REPLACE(REPLACE(rates_residents, 'Ksh ', ''), ' per night', ''), ',', '') AS INTEGER) <= %s)
                OR (rates_non_residents != 'N/A' AND CAST(REPLACE(REPLACE(REPLACE(rates_non_residents, '$', ''), ' per night', ''), ',', '') AS INTEGER) <= %s)
                OR rates_citizens = 'N/A'
            """)
            params.extend([budget, budget, budget])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        for result in results:
            name, loc, description, activities, best_time, citizens, residents, non_residents = result
            rates = {
                "Citizens": citizens,
                "Residents": residents,
                "Non-Residents": non_residents
            }
            attraction = TourismAttraction(
                name=name,
                location=loc,
                description=description,
                activities=activities,
                best_time_to_visit=best_time,
                rates=rates
            ).model_dump()  # Use model_dump() instead of dict()
            attractions.append(attraction)
        
        return attractions
    
    except Exception as e:
        print(f"Error querying database: {e}")
        return attractions
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# LLM Prompt Template
prompt_template = PromptTemplate(
    template="You are a helpful tourism chatbot for Kenya. Use the following information about attractions to answer the user's question. If the information is insufficient, say 'I don’t have enough data to answer that.'\n\nAttractions: {attractions}\n\nUser Question: {user_question}\nAI:",
    input_variables=["attractions", "user_question"]
)

# Chatbot function
def chatbot_response(user_question: str, location: Optional[str] = None, budget: Optional[int] = None):
    # Fetch relevant attractions from the database
    attractions = query_attractions(location, budget)
    
    # Format attractions into a string for the prompt
    attractions_str = "\n".join([f"- {attr['name']}: {attr['description']} (Location: {attr['location']}, Rates: {attr['rates']}, Best Time: {attr['best_time_to_visit']}, Activities: {', '.join(attr['activities'])})" for attr in attractions]) or "No attractions found."
    
    # Create the chain
    chain = prompt_template | llm | output_parser
    
    # Invoke the chain with the attractions and user question
    response = chain.invoke({
        "attractions": attractions_str,
        "user_question": user_question
    })
    
    return response

# Interactive Chatbot Loop
if __name__ == "__main__":
    print("Welcome to the Kenya Tourism Chatbot! Type 'quit' to exit.")
    while True:
        # Prompt user for input
        user_question = input("Ask me about tourism in Kenya: ")
        
        # Exit condition
        if user_question.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Simple parsing for location and budget
        location = None
        budget = None
        question_lower = user_question.lower()
        if "nairobi" in question_lower:
            location = "Nairobi"
        if "ksh" in question_lower:
            # Extract budget (e.g., "under 1000 Ksh" -> 1000)
            words = question_lower.split()
            for i, word in enumerate(words):
                if word == "ksh" and i > 0:
                    try:
                        budget = int(words[i-1])
                        break
                    except ValueError:
                        pass
        
        print(f"Question: {user_question}")
        print(f"Response: {chatbot_response(user_question, location, budget)}")
        print("-" * 50)