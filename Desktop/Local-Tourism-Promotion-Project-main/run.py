# run.py
import sys
sys.path.append("app")  # Add app directory to Python path
from app.routes import app

if __name__ == "__main__":
    app.run(debug=True)