# app/auth.py
from flask import Flask, redirect, url_for, session, render_template, request, flash
from authlib.integrations.flask_client import OAuth
import psycopg2
from psycopg2 import Error
from db_config import DB_CONFIG
import os
import bcrypt
import re
from email_validator import validate_email, EmailNotValidError
from dotenv import load_dotenv
import traceback
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

print("Starting auth.py...")

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a-default-key-for-dev-only')
print("Flask app initialized.")

# Debug: Print environment variables to verify they are loaded
print(f"FLASK_SECRET_KEY: {app.secret_key}")
print(f"AUTH0_DOMAIN: {os.getenv('AUTH0_DOMAIN')}")
print(f"AUTH0_CLIENT_ID: {os.getenv('AUTH0_CLIENT_ID')}")
print(f"AUTH0_CLIENT_SECRET: {os.getenv('AUTH0_CLIENT_SECRET')}")

# Debug: Print session type before OAuth initialization
print(f"Session type before OAuth init: {type(session)}")

oauth = OAuth(app)
print("OAuth initialized.")

# Debug: Print session type after OAuth initialization
print(f"Session type after OAuth init: {type(session)}")

# Load Auth0 credentials from environment variables
auth0_domain = os.getenv('AUTH0_DOMAIN')
auth0_client_id = os.getenv('AUTH0_CLIENT_ID')
auth0_client_secret = os.getenv('AUTH0_CLIENT_SECRET')

# Validate that the environment variables are loaded
if not all([auth0_domain, auth0_client_id, auth0_client_secret]):
    raise ValueError("Missing Auth0 credentials in environment variables. Check your .env file.")

# Manually fetch the OIDC discovery document to debug the issue
print("Fetching OIDC discovery document...")
try:
    discovery_url = f'https://{auth0_domain}/.well-known/openid-configuration'
    response = requests.get(discovery_url, timeout=10)
    response.raise_for_status()
    oidc_metadata = response.json()
    print("OIDC metadata:", oidc_metadata)
    jwks_uri = oidc_metadata.get('jwks_uri')
    print(f"jwks_uri: {jwks_uri}")
except Exception as e:
    print(f"Error fetching OIDC discovery document: {e}")

# Create a custom requests session with retries and timeout
requests_session = requests.Session()  # Renamed to avoid conflict with Flask's session
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
requests_session.mount('https://', HTTPAdapter(max_retries=retries))
app.config['AUTHLIB_CLIENT_REQUESTS_SESSION'] = requests_session

# Register Auth0 with the loaded credentials
auth0 = oauth.register(
    'auth0',
    client_id=auth0_client_id,
    client_secret=auth0_client_secret,
    api_base_url=f'https://{auth0_domain}',
    access_token_url=f'https://{auth0_domain}/oauth/token',
    authorize_url=f'https://{auth0_domain}/authorize',
    jwks_uri=jwks_uri if 'jwks_uri' in locals() else f'https://{auth0_domain}/.well-known/jwks.json',
    client_kwargs={'scope': 'openid profile email', 'timeout': 10},
)
print("Auth0 registered.")

# Validation functions
def validate_full_name(full_name):
    if not full_name or len(full_name.strip()) == 0:
        return False, "Full name cannot be empty."
    if not re.match(r"^[A-Za-z\s]+$", full_name):
        return False, "Full name can only contain letters and spaces."
    return True, ""

def validate_email_format(email):
    try:
        validate_email(email, check_deliverability=False)
        return True, ""
    except EmailNotValidError as e:
        return False, f"Invalid email format: {str(e)}"

def validate_phone(phone):
    if not phone:  # Phone is optional
        return True, ""
    if not re.match(r"^\d{10}$", phone):
        return False, "Phone number must be 10 digits (e.g., 1234567890)."
    return True, ""

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character (e.g., !@#$%^&*)."
    return True, ""

# Function to insert a user into the database
def insert_user(full_name, email, phone, password):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        insert_query = """
        INSERT INTO users (full_name, email, phone, password_hash)
        VALUES (%s, %s, %s, %s);
        """
        cursor.execute(insert_query, (full_name, email, phone, password_hash))
        connection.commit()
        print(f"User {email} inserted successfully.")
        return True
    except Error as e:
        print(f"Error inserting user: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to verify login credentials
def verify_user(email, password):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE email = %s;", (email,))
        result = cursor.fetchone()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            print(f"User {email} verified successfully.")
            return True
        return False
    except Error as e:
        print(f"Error verifying user: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@app.route('/')
def home():
    print("Handling / route...")
    return "Welcome to Pori! <a href='/login'>Sign In</a> | <a href='/signup'>Sign Up</a>"

@app.route('/login', methods=['GET'])
def login_page():
    print("Handling /login route...")
    return render_template('sign_in.html')

@app.route('/login', methods=['POST'])
def login():
    print("Handling /login POST route...")
    email = request.form['email']
    password = request.form['password']
    
    if verify_user(email, password):
        session['email'] = email
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid email or password.', 'error')
        return redirect(url_for('login_page'))

@app.route('/signup', methods=['GET'])
def signup_page():
    print("Handling /signup route...")
    return render_template('sign_up.html')

@app.route('/register', methods=['POST'])
def register():
    print("Handling /register POST route...")
    full_name = request.form['full_name']
    email = request.form['email']
    phone = request.form.get('phone', '')
    password = request.form['password']

    # Validate form data
    is_valid, full_name_error = validate_full_name(full_name)
    if not is_valid:
        flash(full_name_error, 'error')
        return redirect(url_for('signup_page'))

    is_valid, email_error = validate_email_format(email)
    if not is_valid:
        flash(email_error, 'error')
        return redirect(url_for('signup_page'))

    is_valid, phone_error = validate_phone(phone)
    if not is_valid:
        flash(phone_error, 'error')
        return redirect(url_for('signup_page'))

    is_valid, password_error = validate_password(password)
    if not is_valid:
        flash(password_error, 'error')
        return redirect(url_for('signup_page'))

    if insert_user(full_name, email, phone, password):
        flash('Registration successful! Please sign in.', 'success')
        return redirect(url_for('login_page'))
    else:
        flash('Registration failed. Email might already exist.', 'error')
        return redirect(url_for('signup_page'))

@app.route('/auth0_login')
def auth0_login():
    print("Handling /auth0_login route...")
    print(f"Session type in /auth0_login: {type(session)}")
    print(f"Session contents before: {session}")
    import secrets
    nonce = secrets.token_urlsafe(16)
    session['nonce'] = nonce
    redirect_uri = url_for('auth0_callback', _external=True)
    print(f"Using redirect_uri: {redirect_uri}")
    print(f"Session contents after: {session}")
    return auth0.authorize_redirect(
        redirect_uri=redirect_uri,
        nonce=nonce
    )

@app.route('/auth0_signup')
def auth0_signup():
    print("Handling /auth0_signup route...")
    print(f"Session type in /auth0_signup: {type(session)}")
    print(f"Session contents before: {session}")
    import secrets
    nonce = secrets.token_urlsafe(16)
    session['nonce'] = nonce
    redirect_uri = url_for('auth0_callback', _external=True)
    print(f"Using redirect_uri: {redirect_uri}")
    print(f"Session contents after: {session}")
    return auth0.authorize_redirect(
        redirect_uri=redirect_uri,
        nonce=nonce
    )

@app.route('/auth0_callback')
def auth0_callback():
    print("Handling /auth0_callback route...")
    try:
        print("Attempting to authorize access token...")
        token = auth0.authorize_access_token()
        print("Access token received:", token)
        
        print("Parsing ID token...")
        nonce = session.get('nonce')
        print(f"Retrieved nonce from session: {nonce}")
        if not nonce:
            raise ValueError("Nonce not found in session. Ensure it was set during authorize_redirect.")
        
        user_info = auth0.parse_id_token(token, nonce=nonce)
        print("User info:", user_info)
        
        auth0_id = user_info['sub']
        email = user_info['email']
        print(f"Auth0 ID: {auth0_id}, Email: {email}")

        print("Connecting to database...")
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        try:
            print("Inserting user into database...")
            cursor.execute("""
                INSERT INTO users (email, auth0_id)
                VALUES (%s, %s)
                ON CONFLICT (email) DO UPDATE SET auth0_id = EXCLUDED.auth0_id
            """, (email, auth0_id))
            connection.commit()
            print(f"Auth0 user {email} saved successfully.")
        except Error as e:
            print(f"Error saving user to database: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

        session['user'] = user_info
        print("Redirecting to dashboard...")
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error in Auth0 callback: {e}")
        print("Stack trace:")
        print(traceback.format_exc())
        flash('Failed to sign up with Google. Please try again.', 'error')
        return redirect(url_for('signup_page'))

@app.route('/logout')
def logout():
    print("Handling /logout route...")
    session.pop('user', None)
    session.pop('email', None)
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    print("Handling /dashboard route...")
    if 'user' in session:
        return f"Welcome, {session['user']['email']}!"
    elif 'email' in session:
        return f"Welcome, {session['email']}!"
    return redirect(url_for('login_page'))

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
    print("Flask server started.")