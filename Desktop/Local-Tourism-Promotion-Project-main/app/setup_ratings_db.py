import psycopg2
from psycopg2 import Error
from db_config import DB_CONFIG

print("Starting attraction_ratings database setup...")

def create_connection(db_key='kenya_tourism'):
    print(f"Attempting to connect to PostgreSQL database: {db_key}...")
    try:
        connection = psycopg2.connect(**DB_CONFIG[db_key])
        print(f"Connected to {db_key} database successfully!")
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def setup_ratings_database():
    connection = None
    cursor = None
    try:
        connection = create_connection('kenya_tourism')
        if not connection:
            print("Failed to establish connection. Exiting.")
            return
        
        cursor = connection.cursor()
        print("Cursor created, executing table creation query...")
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attraction_ratings (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,  -- Matches your session['user_id'] format
            attraction_name VARCHAR(255) NOT NULL,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id, attraction_name)  -- One rating per user per attraction
        );
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("attraction_ratings table created successfully")
        
    except Error as e:
        print(f"Error setting up attraction_ratings database: {e}")
    finally:
        if cursor is not None:
            cursor.close()
            print("Cursor closed")
        if connection is not None:
            connection.close()
            print("Connection closed")

if __name__ == "__main__":
    print("Running setup_ratings_database directly...")
    setup_ratings_database()