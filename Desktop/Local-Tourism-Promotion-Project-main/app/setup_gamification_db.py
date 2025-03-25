import psycopg2
from psycopg2 import Error
from db_config import DB_CONFIG

print("Starting gamification database setup...")

def create_connection():
    print("Attempting to connect to PostgreSQL for gamification setup...")
    try:
        # Use the 'kenya_tourism' database configuration
        connection = psycopg2.connect(**DB_CONFIG['kenya_tourism'])
        print("Connected to PostgreSQL successfully!")
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def setup_gamification_database():
    connection = None
    cursor = None
    try:
        # Create connection
        connection = create_connection()
        if not connection:
            print("Failed to establish connection. Exiting.")
            return
        
        cursor = connection.cursor()
        print("Cursor created, executing gamification table creation query...")
        
        # Create user_gamification table
        create_gamification_table_query = """
        CREATE TABLE IF NOT EXISTS user_gamification (
            user_id VARCHAR(32) PRIMARY KEY,
            points INTEGER DEFAULT 0,
            badges TEXT[] DEFAULT '{}'
        );
        """
        
        cursor.execute(create_gamification_table_query)
        connection.commit()
        print("User_gamification table created successfully")
        
    except Error as e:
        print(f"Error setting up gamification database: {e}")
    finally:
        if cursor is not None:
            cursor.close()
            print("Cursor closed")
        if connection is not None:
            connection.close()
            print("Connection closed")

if __name__ == "__main__":
    print("Running setup_gamification_database directly...")
    setup_gamification_database()