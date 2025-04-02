# app/setup_auth_users_db.py
import psycopg2
from psycopg2 import Error
from db_config import DB_CONFIG

print("Starting auth_users database setup...")

def create_connection(db_key='auth_users'):
    print(f"Attempting to connect to PostgreSQL database: {db_key}...")
    try:
        connection = psycopg2.connect(**DB_CONFIG[db_key])
        print(f"Connected to {db_key} database successfully!")
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def setup_auth_users_database():
    connection = None
    cursor = None
    try:
        connection = create_connection('auth_users')
        if not connection:
            print("Failed to establish connection. Exiting.")
            return
        
        cursor = connection.cursor()
        print("Cursor created, executing table creation query...")
        
        # Create users table if it doesn’t exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            full_name VARCHAR(100),
            email VARCHAR(255) UNIQUE NOT NULL,
            phone VARCHAR(20),
            password_hash VARCHAR(255),
            auth0_id VARCHAR(255) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        cursor.execute(create_table_query)
        print("Ensured users table exists.")

        # Check if is_admin column exists, and add it if not
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'users' AND column_name = 'is_admin';
        """)
        if not cursor.fetchone():
            print("is_admin column not found, adding it...")
            alter_table_query = """
            ALTER TABLE users 
            ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
            """
            cursor.execute(alter_table_query)
            print("is_admin column added successfully.")
        else:
            print("is_admin column already exists, no changes needed.")

        connection.commit()
        print("auth_users table setup completed successfully.")
        
    except Error as e:
        print(f"Error setting up auth_users database: {e}")
    finally:
        if cursor is not None:
            cursor.close()
            print("Cursor closed")
        if connection is not None:
            connection.close()
            print("Connection closed")

if __name__ == "__main__":
    print("Running setup_auth_users_database directly...")
    setup_auth_users_database()