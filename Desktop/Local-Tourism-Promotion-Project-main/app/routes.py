from flask import Flask, request, jsonify, render_template
import requests
from app.db_config import DB_CONFIG
import psycopg2
import os
from rag import ask_question  # Import the RAG query function

app = Flask(__name__, template_folder='frontend/templates', static_folder='static')

# Database connection
def create_connection():
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("Database connection successful")
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

@app.route('/')
def index():
    try:
        return render_template('tourist_filter.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return jsonify({"error": "Template rendering failed"}), 500

@app.route('/tourist_filter')
def tourist_filter():
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY', 'default_key_if_not_set')
        return render_template('tourist_filter.html', openweather_api_key=api_key)
    except Exception as e:
        print(f"Error rendering tourist_filter template: {e}")
        return jsonify({"error": "Template rendering failed"}), 500
    
@app.route('/get_weather')
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return jsonify({"error": "API key not found"}), 500
    response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric')
    if response.status_code != 200:
        return jsonify({"error": f"Weather API error: {response.status_code} - {response.text}"}), response.status_code
    return jsonify(response.json())    

@app.route('/filter_attractions', methods=['POST'])
def filter_attractions():
    filters = request.get_json()
    location = filters.get('location')
    budget = filters.get('budget')
    experience = filters.get('experience')

    print(f"Received filters: {filters}")

    # Parse budget range
    budget_min = None
    budget_max = None
    if budget:
        if budget == "10000+":
            budget_min = 10000
            budget_max = 100000
        else:
            budget_range = budget.split('-')
            budget_min = int(budget_range[0])
            budget_max = int(budget_range[1])

    connection = create_connection()
    if not connection:
        print("Database connection failed")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()
        query = """
        SELECT name, location, description, activities, best_time_to_visit, 
               rates_citizens, rates_residents, rates_non_residents
        FROM attractions
        WHERE (%s IS NULL OR location ILIKE %s)
        """
        params = [location, f"%{location}%" if location else None]

        print(f"Executing base query with params: {params}")
        cursor.execute(query, params)
        results = cursor.fetchall()
        print(f"Base query returned {len(results)} results: {results}")

        location_coords = {
            "maasai-mara": {"lat": -1.5000, "lon": 35.0000},
            "nairobi": {"lat": -1.2921, "lon": 36.8219},
            "mombasa": {"lat": -4.0433, "lon": 39.6682},
            "lamu": {"lat": -2.2717, "lon": 40.9020},
            "nakuru": {"lat": -0.3031, "lon": 36.0800},
            "kajiado": {"lat": -1.8457, "lon": 36.7850},
            "homa-bay": {"lat": -0.5273, "lon": 34.4571}
        }

        attractions = []
        for result in results:
            name, loc, desc, activities, best_time, citizens, residents, non_res = result
            coords = location_coords.get(loc.lower(), {"lat": 0, "lon": 0})
            rate_value = int(citizens.replace('Ksh ', '').replace(',', '')) if citizens != 'N/A' else 0

            if budget_min is not None and (rate_value < budget_min or rate_value > budget_max):
                continue
            if experience and experience not in activities:
                continue

            attractions.append({
                "name": name,
                "location": loc,
                "description": desc,
                "experience": activities,
                "best_time_to_visit": best_time,
                "rates_citizens": citizens,
                "rates_residents": residents,
                "rates_non_residents": non_res,
                "lat": coords["lat"],
                "lon": coords["lon"]
            })

        print(f"Final filtered results: {len(attractions)} attractions: {attractions}")
        return jsonify({"attractions": attractions})

    except Exception as e:
        print(f"Error querying database: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# New Chatbot Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        response = ask_question(query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)