import psycopg2
from psycopg2 import Error
from .db_config import DB_CONFIG

print("Starting database setup...")  # Initial debug message

def create_connection():
    print("Attempting to connect to PostgreSQL...")
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL successfully!")
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def setup_database():
    connection = None
    cursor = None
    try:
        # Create connection
        connection = create_connection()
        if not connection:
            print("Failed to establish connection. Exiting.")
            return
        
        cursor = connection.cursor()
        print("Cursor created, executing table creation query...")
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attractions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            location VARCHAR(100) NOT NULL,
            description TEXT,
            activities TEXT[],
            best_time_to_visit VARCHAR(50),
            rates_citizens VARCHAR(20),
            rates_residents VARCHAR(20),
            rates_non_residents VARCHAR(20)
        );
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("Database table created successfully")
        
    except Error as e:
        print(f"Error setting up database: {e}")
    finally:
        if cursor is not None:
            cursor.close()
            print("Cursor closed")
        if connection is not None:
            connection.close()
            print("Connection closed")

if __name__ == "__main__":
    print("Running setup_database directly...")
    setup_database()