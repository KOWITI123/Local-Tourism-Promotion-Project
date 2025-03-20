from flask import Flask, render_template, session, redirect, url_for
from app.auth import auth
from authentication.db import db
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Initialize DB
db.init_app(app)
with app.app_context():
    db.create_all()

# Register Authentication Blueprint
app.register_blueprint(auth)

@app.route('/')
def home():
    return render_template('sign_in.html')  # Default to Sign In page

@app.route('/sign_up')
def sign_up():
    return render_template('sign_up.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html', user=session['user'])
    return "Access Denied", 403

if __name__ == '__main__':
    app.run(debug=True)
