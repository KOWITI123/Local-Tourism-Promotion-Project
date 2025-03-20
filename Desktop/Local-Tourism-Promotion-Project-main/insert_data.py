# insert_test_data.py
from app.db_config import DB_CONFIG
import psycopg2

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attractions (name, location, description, activities, best_time_to_visit, rates_citizens, rates_residents, rates_non_residents)
        VALUES 
        ('Maasai Mara National Reserve', 'maasai-mara', 'Famous for the Great Migration', '{"Wildlife Safari", "Photography"}', 'July-October', 'Ksh 1000', 'Ksh 2000', '$80'),
        ('Nairobi National Park', 'nairobi', 'Wildlife near the city', '{"Safari", "Bird Watching"}', 'All Year', 'Ksh 500', 'Ksh 1000', '$50'),
        ('Diani Beach', 'mombasa', 'Beautiful coastal beach', '{"Swimming", "Relaxation"}', 'December-March', 'Ksh 0', 'Ksh 0', '$0')
    """)
    conn.commit()
    print("Test data inserted successfully")
except psycopg2.Error as e:
    print(f"Error inserting data: {e}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()