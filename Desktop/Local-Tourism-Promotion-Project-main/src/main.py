from ..app.db_setup import setup_database  # Relative import
from .models.attraction import Attraction      # Relative import

def main():
    setup_database()
    attraction_manager = Attraction()
    data_path = r'C:\Users\user\Desktop\Local-Tourism-Promotion-Project\data\tourist_sites.json'
    attraction_manager.load_json_data(data_path)

if __name__ == "__main__":
    main()