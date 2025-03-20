import json
import psycopg2
from ...app.db_setup import create_connection

class Attraction:
    def __init__(self):
        self.connection = None
        
    def insert_attraction(self, attraction_data):
        try:
            self.connection = create_connection()
            cursor = self.connection.cursor()
            
            insert_query = """
            INSERT INTO attractions (
                name, location, description, activities, 
                best_time_to_visit, rates_citizens, 
                rates_residents, rates_non_residents
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            # Prepare data for insertion
            rates = attraction_data.get('rates', {})
            # Check if rates is a string (e.g., "N/A")
            if isinstance(rates, str):
                # If rates is "N/A", set all rate fields to "N/A"
                rates_citizens = rates_residents = rates_non_residents = rates
            else:
                # If rates is a dictionary, use .get() to extract values
                rates_citizens = rates.get('Citizens', 'N/A')
                rates_residents = rates.get('Residents', 'N/A')
                rates_non_residents = rates.get('Non-Residents', 'N/A')
            
            values = (
                attraction_data['name'],
                attraction_data['location'],
                attraction_data['description'],
                attraction_data['activities'],
                attraction_data['best_time_to_visit'],
                rates_citizens,
                rates_residents,
                rates_non_residents
            )
            
            cursor.execute(insert_query, values)
            self.connection.commit()
            print(f"Inserted {attraction_data['name']} successfully")
            
        except Exception as e:
            print(f"Error inserting attraction: {e}")
        finally:
            if cursor:
                cursor.close()
            if self.connection:
                self.connection.close()

    def load_json_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            for attraction in data:
                self.insert_attraction(attraction)