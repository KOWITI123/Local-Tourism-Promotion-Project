import secrets
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, current_user, login_required
import requests
from app.db_config import DB_CONFIG
import psycopg2
import os
from rag import ask_question
import hashlib  # Required for signature generation
import time     # Required for timestamp
from datetime import datetime
from dotenv import load_dotenv
# Google Calendar API imports
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
# Auth0 imports
from authlib.integrations.flask_client import OAuth
import bcrypt
import re
from email_validator import validate_email, EmailNotValidError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import wraps

from googleapiclient.discovery import build
from googleapiclient.errors import UnknownApiNameOrVersion

load_dotenv()

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development
print(f"Secret key set to: {app.secret_key}")

app.config['SERVER_NAME'] = '127.0.0.1:5000'  
app.config['SESSION_COOKIE_DOMAIN'] = None   
app.config['SESSION_COOKIE_SECURE'] = False   # For local testing (HTTP, not HTTPS)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' # Ensure cookies work across redirects

# Auth0 Setup
oauth = OAuth(app)
print("OAuth initialized.")

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
requests_session = requests.Session()
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

# Google Calendar API Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', 'your_client_id')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', 'your_client_secret')
SCOPES = ['https://www.googleapis.com/auth/calendar.events']
REDIRECT_URI = 'http://127.0.0.1:5000/auth/google/callback'

# Hotelbeds API Configuration
HOTELBEDS_API_KEY = os.getenv('HOTELBEDS_API_KEY', 'your_api_key')
HOTELBEDS_SECRET = os.getenv('HOTELBEDS_SECRET', 'your_secret')
HOTELBEDS_BASE_URL = 'https://api.test.hotelbeds.com/hotel-api/1.0'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'google_auth'  # Redirect to Google auth instead of /login

class User(UserMixin):
    def __init__(self, user_id, role='user'):
        self.id = user_id
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        cursor.execute("SELECT id, role FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            print(f"Loaded user: id={result[0]}, role={result[1]}")
            return User(result[0], result[1] or 'user')
        print(f"No user found for id={user_id}")
        return None
    except psycopg2.Error as e:
        print(f"Error loading user: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Validation functions (from app/auth.py)
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

# Function to insert a user into the database (from app/auth.py)
def insert_user(full_name, email, phone, password):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        insert_query = """
        INSERT INTO users (full_name, email, phone, password_hash)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        cursor.execute(insert_query, (full_name, email, phone, password_hash))
        user_id = cursor.fetchone()[0]
        connection.commit()
        print(f"User {email} inserted successfully with user_id: {user_id}.")
        return True, user_id
    except psycopg2.Error as e:
        print(f"Error inserting user: {e}")
        return False, None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to verify login credentials (from app/auth.py)
def verify_user(email, password):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        cursor.execute("SELECT id, password_hash, role FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
            print(f"User {email} verified successfully.")
            return True, result[0], result[2] or 'user'  # Return user_id and role
        return False, None, None
    except psycopg2.Error as e:
        print(f"Error verifying user: {e}")
        return False, None, None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def create_connection():
    try:
        connection = psycopg2.connect(**DB_CONFIG['kenya_tourism'])
        print("Database connection successful")
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def get_user_id():
    if 'user_id' not in session:
        # If the user logged in via Auth0 or manual login, user_id should be set
        if 'user' in session:
            # For Auth0 users, fetch user_id from the database using email
            email = session['user']['email']
            connection = create_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    result = cursor.fetchone()
                    if result:
                        session['user_id'] = result[0]
                    else:
                        # If user doesn't exist in DB (unlikely), generate a new user_id
                        session['user_id'] = os.urandom(16).hex()
                finally:
                    cursor.close()
                    connection.close()
        elif 'email' in session:
            # For manual login users, fetch user_id using email
            email = session['email']
            connection = create_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    result = cursor.fetchone()
                    if result:
                        session['user_id'] = result[0]
                    else:
                        session['user_id'] = os.urandom(16).hex()
                finally:
                    cursor.close()
                    connection.close()
        else:
            # Fallback: Generate a new user_id
            session['user_id'] = os.urandom(16).hex()
        print(f"New user_id generated: {session['user_id']}")
    return session['user_id']

def add_points(user_id, points):
    connection = create_connection()
    if not connection:
        print("Failed to connect to database for adding points")
        return False
    try:
        cursor = connection.cursor()
        print(f"Adding {points} points for user_id: {user_id}")
        cursor.execute("""
            INSERT INTO user_gamification (user_id, points)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET points = user_gamification.points + %s
        """, (user_id, points, points))
        connection.commit()
        print(f"Points added successfully for user_id: {user_id}")
        return True
    except Exception as e:
        print(f"Error adding points: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def get_user_stats(user_id):
    connection = create_connection()
    if not connection:
        print("Failed to connect for stats retrieval")
        return {"points": 0, "badges": [], "level": 1}
    try:
        cursor = connection.cursor()
        print(f"Fetching stats for user_id: {user_id}")
        cursor.execute("SELECT points FROM user_gamification WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        points = result[0] if result else 0
        level = 1 + (points // 100)
        badges = get_user_badges(user_id, points)
        print(f"Stats for {user_id}: points={points}, level={level}, badges={badges}")
        return {"points": points, "badges": badges, "level": level}
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"points": 0, "badges": [], "level": 1}
    finally:
        cursor.close()
        connection.close()

def get_user_badges(user_id, points):
    badges = []
    if points >= 10:
        badges.append("First Step")
    if points >= 50:
        badges.append("Explorer")
    return badges

def generate_signature():
    timestamp = str(int(time.time()))
    signature = hashlib.sha256(f"{HOTELBEDS_API_KEY}{HOTELBEDS_SECRET}{timestamp}".encode()).hexdigest()
    return signature, timestamp

# Helper function to check if a user is logged in

def login_required(f):
    def wrap(*args, **kwargs):
        if 'user' not in session and 'email' not in session:
            flash("Please sign in to access this page.", "error")
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        print(f"Checking admin access: is_authenticated={current_user.is_authenticated}, role={getattr(current_user, 'role', 'None')}, user_id={getattr(current_user, 'id', 'None')}")
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('You must be an admin to access this page.', 'error')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication Routes (from app/auth.py)
@app.route('/login', methods=['GET'])
def login_page():
    print("Handling /login route...")
    return render_template('sign_in.html')

@app.route('/login', methods=['POST'])
def login():
    print("Handling /login POST route...")
    email = request.form['email']
    password = request.form['password']
    
    success, user_id, role = verify_user(email, password)
    if success:
        session['email'] = email
        session['user_id'] = user_id
        user = User(user_id, role)  # Pass role to User
        login_user(user)
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
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

    success, user_id = insert_user(full_name, email, phone, password)
    if success:
        session['user_id'] = user_id
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

    # Only clear session if starting fresh
    if 'nonce' not in session or 'user' not in session:
        session.clear()

    nonce = secrets.token_urlsafe(16)
    session['nonce'] = nonce
    redirect_uri = url_for('auth0_callback', _external=True)
    
    referrer = request.referrer or ''
    if 'admin_login' in referrer:
        session['login_destination'] = 'admin'
        print("Set login_destination to 'admin'")
    
    print(f"Using redirect_uri: {redirect_uri}")
    print(f"Session contents after: {session}")
    return auth0.authorize_redirect(
        redirect_uri=redirect_uri,
        nonce=nonce,
        prompt='login select_account',  # Force login and account chooser
        login_hint=None,
        max_age=0
    )
    

@app.route('/auth0_signup')
def auth0_signup():
    print("Handling /auth0_signup route...")
    print(f"Session type in /auth0_signup: {type(session)}")
    print(f"Session contents before: {session}")
    import secrets
    nonce = secrets.token_urlsafe(16)
    session['nonce'] = nonce
    session['signup_intent'] = True
    
    # Log out of Auth0 to clear any existing session
    logout_url = f"https://{auth0_domain}/v2/logout?client_id={auth0_client_id}&returnTo={url_for('signup_page', _external=True)}"
    requests.get(logout_url)  # Perform a logout request to Auth0
    
    redirect_uri = url_for('auth0_callback', _external=True)
    print(f"Using redirect_uri: {redirect_uri}")
    print(f"Session contents after: {session}")
    return auth0.authorize_redirect(
        redirect_uri=redirect_uri,
        nonce=nonce,
        screen_hint='signup',          # Encourage signup screen
        prompt='login select_account'  # Force re-authentication AND account chooser
    )

@app.route('/auth0_callback')
def auth0_callback():
    print("Handling /auth0_callback route...")
    try:
        token = auth0.authorize_access_token()
        nonce = session.get('nonce')
        if not nonce:
            raise ValueError("Nonce not found in session.")
        user_info = auth0.parse_id_token(token, nonce=nonce)
        auth0_id = user_info['sub']
        email = user_info['email']
        print(f"Auth0 ID: {auth0_id}, Email: {email}")

        connection = psycopg2.connect(**DB_CONFIG['auth_users'])
        cursor = connection.cursor()
        
        cursor.execute("SELECT id, role FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            user_id, role = existing_user
            if role != 'admin' and 'signup' in (request.referrer or ''):
                cursor.close()
                connection.close()
                flash('This email is already registered. Please sign in instead.', 'error')
                return redirect(url_for('login_page'))
        else:
            cursor.execute("""
                INSERT INTO users (email, auth0_id, role)
                VALUES (%s, %s, 'user')
                RETURNING id, role
            """, (email, auth0_id))
            user_id, role = cursor.fetchone()
            connection.commit()
            print(f"Auth0 user {email} saved with user_id: {user_id}, role: {role}.")
        
        cursor.close()
        connection.close()

        session['user'] = user_info
        session['user_id'] = str(user_id)
        user = User(user_id, role)
        login_user(user, remember=True)
        print(f"User logged in: id={user_id}, role={role}")
        
        login_destination = session.pop('login_destination', None)
        if role == 'admin' and login_destination == 'admin':
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_page'))  # Changed to admin_page
        else:
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

    except Exception as e:
        print(f"Error in Auth0 callback: {e}")
        import traceback
        print(traceback.format_exc())
        flash('Failed to sign in with Google. Please try again.', 'error')
        return redirect(url_for('login_page'))
    
# Admin login POST
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    print("Handling /admin_login route...")
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        success, user_id, role = verify_user(email, password)
        if success and role == 'admin':
            session['email'] = email
            session['user_id'] = str(user_id)
            user = User(user_id, role)
            login_user(user, remember=True)
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_page'))  # Changed to admin_page
        else:
            flash('Invalid admin credentials or not an admin account.', 'error')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

# Admin dashboard route
@app.route('/admin_page')
@admin_required
def admin_page():
    print("Handling /admin_page route...")
    connection = create_connection()  # Use 'kenya_tourism' DB for attractions
    if not connection:
        return render_template('admin.html', total_users=0, active_sessions=0, content_items=0, users=[])
    
    try:
        cursor = connection.cursor()
        # Total users from auth_users DB
        auth_conn = psycopg2.connect(**DB_CONFIG['auth_users'])
        auth_cursor = auth_conn.cursor()
        auth_cursor.execute("SELECT COUNT(*) FROM users;")
        total_users = auth_cursor.fetchone()[0]
        
        # Active sessions (simplified as users with is_active = TRUE)
        auth_cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE;")
        active_sessions = auth_cursor.fetchone()[0]
        
        # Content items from attractions table
        cursor.execute("SELECT COUNT(*) FROM attractions;")
        content_items = cursor.fetchone()[0]
        
        # Fetch user list
        auth_cursor.execute("SELECT id, full_name, email FROM users LIMIT 10;")
        users = [{'id': row[0], 'full_name': row[1], 'email': row[2]} for row in auth_cursor.fetchall()]

        # Fetch attractions
        cursor.execute("SELECT id, name, location FROM attractions LIMIT 10;")
        attractions = [{'id': row[0], 'name': row[1], 'location': row[2]} for row in cursor.fetchall()]
        
    finally:
        cursor.close()
        connection.close()
        auth_cursor.close()
        auth_conn.close()
    
    return render_template('admin.html', total_users=total_users, active_sessions=active_sessions, content_items=content_items, users=users, attractions=attractions)

# Admin add user
@app.route('/admin/add_user', methods=['POST'])
@admin_required
def admin_add_user():
    full_name = request.form['full_name']
    email = request.form['email']
    password = request.form['password']
    
    success, user_id = insert_user(full_name, email, '', password)  # Phone is optional
    if success:
        flash('User added successfully!', 'success')
    else:
        flash('Failed to add user.', 'error')
    return redirect(url_for('admin_page'))

# Admin delete user
@app.route('/admin/delete_user/<int:user_id>')
@admin_required
def admin_delete_user(user_id):
    connection = psycopg2.connect(**DB_CONFIG['auth_users'])
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM users WHERE id = %s AND role != 'admin';", (user_id,))  # Prevent deleting admins
            connection.commit()
            if cursor.rowcount > 0:
                flash('User deleted successfully!', 'success')
            else:
                flash('User not found or is an admin.', 'error')
        except Exception as e:
            flash(f'Error deleting user: {e}', 'error')
        finally:
            cursor.close()
            connection.close()
    return redirect(url_for('admin_page'))

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    connection = psycopg2.connect(**DB_CONFIG['auth_users'])
    cursor = connection.cursor()
    
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form.get('password', '')  # Optional
        
        try:
            if password:
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("""
                    UPDATE users SET full_name = %s, email = %s, password_hash = %s
                    WHERE id = %s AND role != 'admin';
                """, (full_name, email, password_hash, user_id))
            else:
                cursor.execute("""
                    UPDATE users SET full_name = %s, email = %s
                    WHERE id = %s AND role != 'admin';
                """, (full_name, email, user_id))
            connection.commit()
            if cursor.rowcount > 0:
                flash('User updated successfully!', 'success')
            else:
                flash('User not found or is an admin.', 'error')
        except Exception as e:
            flash(f'Error updating user: {e}', 'error')
        finally:
            cursor.close()
            connection.close()
        return redirect(url_for('admin_page'))
    
    # GET: Fetch user data
    try:
        cursor.execute("SELECT full_name, email FROM users WHERE id = %s AND role != 'admin';", (user_id,))
        user = cursor.fetchone()
        if user:
            return render_template('edit_user.html', user={'id': user_id, 'full_name': user[0], 'email': user[1]})
        flash('User not found or is an admin.', 'error')
        return redirect(url_for('admin_page'))
    finally:
        cursor.close()
        connection.close()

# Admin add content (simplified, assumes attractions table)
@app.route('/admin/add_content', methods=['POST'])
@admin_required
def admin_add_content():
    name = request.form['name']
    location = request.form['location']
    description = request.form['description']
    activities = request.form['activities']
    best_time_to_visit = request.form['best_time_to_visit']
    rates_citizens = request.form['rates_citizens']
    rates_residents = request.form['rates_residents']
    rates_non_residents = request.form['rates_non_residents']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO attractions (name, location, description, activities, best_time_to_visit, 
                                        rates_citizens, rates_residents, rates_non_residents, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (name, location, description, activities, best_time_to_visit, rates_citizens, 
                  rates_residents, rates_non_residents, latitude, longitude))
            connection.commit()
            flash('Attraction added successfully!', 'success')
        except Exception as e:
            flash(f'Error adding attraction: {e}', 'error')
        finally:
            cursor.close()
            connection.close()
    return redirect(url_for('admin_page'))

@app.route('/admin/edit_attraction/<int:attraction_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_attraction(attraction_id):
    connection = create_connection()
    cursor = connection.cursor()
    
    if request.method == 'POST':
        name = request.form['name']
        location = request.form['location']
        description = request.form['description']
        activities = request.form['activities']
        best_time_to_visit = request.form['best_time_to_visit']
        rates_citizens = request.form['rates_citizens']
        rates_residents = request.form['rates_residents']
        rates_non_residents = request.form['rates_non_residents']
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        
        try:
            cursor.execute("""
                UPDATE attractions 
                SET name = %s, location = %s, description = %s, activities = %s, best_time_to_visit = %s,
                    rates_citizens = %s, rates_residents = %s, rates_non_residents = %s, latitude = %s, longitude = %s
                WHERE id = %s;
            """, (name, location, description, activities, best_time_to_visit, rates_citizens, 
                  rates_residents, rates_non_residents, latitude, longitude, attraction_id))
            connection.commit()
            if cursor.rowcount > 0:
                flash('Attraction updated successfully!', 'success')
            else:
                flash('Attraction not found.', 'error')
        except Exception as e:
            flash(f'Error updating attraction: {e}', 'error')
        finally:
            cursor.close()
            connection.close()
        return redirect(url_for('admin_page'))
    
    # GET: Fetch attraction data
    try:
        cursor.execute("""
            SELECT id, name, location, description, activities, best_time_to_visit, 
                   rates_citizens, rates_residents, rates_non_residents, latitude, longitude 
            FROM attractions WHERE id = %s;
        """, (attraction_id,))
        attraction = cursor.fetchone()
        if attraction:
            attraction_data = {
                'id': attraction[0], 'name': attraction[1], 'location': attraction[2], 
                'description': attraction[3], 'activities': attraction[4], 'best_time_to_visit': attraction[5],
                'rates_citizens': attraction[6], 'rates_residents': attraction[7], 
                'rates_non_residents': attraction[8], 'latitude': attraction[9], 'longitude': attraction[10]
            }
            return render_template('edit_attraction.html', attraction=attraction_data)
        flash('Attraction not found.', 'error')
        return redirect(url_for('admin_page'))
    finally:
        cursor.close()
        connection.close()

@app.route('/admin/delete_attraction/<int:attraction_id>')
@admin_required
def admin_delete_attraction(attraction_id):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM attractions WHERE id = %s;", (attraction_id,))
            connection.commit()
            if cursor.rowcount > 0:
                flash('Attraction deleted successfully!', 'success')
            else:
                flash('Attraction not found.', 'error')
        except Exception as e:
            flash(f'Error deleting attraction: {e}', 'error')
        finally:
            cursor.close()
            connection.close()
    return redirect(url_for('admin_page'))

@app.route('/logout')
def logout():
    print("Handling /logout route...")
    # Clear Flask session for both manual and Auth0 logins
    session.pop('user', None)      # Auth0 user info
    session.pop('email', None)     # Manual login email
    session.pop('user_id', None)   # User ID
    session.pop('nonce', None)     # Auth0 nonce
    session.pop('signup_intent', None)  # Signup intent flag
    session.pop('google_credentials', None)  # Google Calendar credentials
    session.pop('is_admin', None)  # Clear admin status
    
    # Optional: Log out of Auth0 to clear its session
    logout_url = f"https://{auth0_domain}/v2/logout?client_id={auth0_client_id}&returnTo={url_for('login_page', _external=True)}"
    requests.get(logout_url)
    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login_page'))    

# Google OAuth 2.0 Routes (for Google Calendar)
@app.route('/auth/google')
@login_required
def google_auth():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
                "scopes": SCOPES
            }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = REDIRECT_URI
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)

@app.route('/auth/google/callback')
def google_callback():
    try:
        state = session.get('state')
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [url_for('google_callback', _external=True)],
                    "scopes": SCOPES
                }
            },
            scopes=SCOPES,
            state=state
        )
        flow.redirect_uri = url_for('google_callback', _external=True)
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        session['google_credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        user_id = session.get('user_id', os.urandom(16).hex())  # Use existing or generate
        session['user_id'] = user_id
        login_user(User(user_id))  # Log in the user for Flask-Login
        print(f"User {user_id} logged in with Google credentials")
        return redirect(url_for('tourist_filter', auth_success=True))
    except Exception as e:
        print(f"Error in Google callback: {e}")
        return redirect(url_for('tourist_filter', auth_error=str(e)))


@app.route('/add_to_calendar', methods=['POST'])
@login_required
def add_to_calendar():
    data = request.get_json()
    print(f"Session contents: {session}")
    if not data:
        return jsonify({"error": "No data provided"}), 400

    attraction_name = data.get('attraction_name')
    visit_date = data.get('visit_date')
    location = data.get('location')

    if not attraction_name or not visit_date or not location:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        datetime.strptime(visit_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    if 'google_credentials' not in session:
        print("Google credentials missing, redirecting to auth")
        return jsonify({"error": "User not authenticated with Google.", "auth_url": url_for('google_auth', _external=True)}), 401

    credentials_dict = session['google_credentials']
    try:
        credentials = Credentials(
            token=credentials_dict['token'],
            refresh_token=credentials_dict['refresh_token'],
            token_uri=credentials_dict['token_uri'],
            client_id=credentials_dict['client_id'],
            client_secret=credentials_dict['client_secret'],
            scopes=credentials_dict['scopes']
        )
    except Exception as e:
        print(f"Error creating credentials: {e}")
        return jsonify({"error": f"Invalid Google credentials: {str(e)}"}), 500

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(google.auth.transport.requests.Request())
        except Exception as e:
            print(f"Error refreshing credentials: {e}")
            return jsonify({"error": f"Failed to refresh Google credentials: {str(e)}"}), 500

    # Add scope check
    print(f"Credentials scopes: {credentials.scopes}")
    if 'https://www.googleapis.com/auth/calendar.events' not in credentials.scopes:
        return jsonify({"error": "Missing Calendar scope. Please re-authenticate.", "auth_url": url_for('google_auth', _external=True)}), 401

    try:
        # Use cache_discovery=False to avoid fetching issues
        service = build('calendar', 'v3', credentials=credentials, cache_discovery=False)
    except UnknownApiNameOrVersion as e:
        print(f"Failed to build Calendar service: {e}")
        return jsonify({"error": f"Google Calendar API unavailable: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error building service: {e}")
        return jsonify({"error": f"Failed to initialize Calendar service: {str(e)}"}), 500

    event = {
        'summary': f'Visit to {attraction_name}',
        'location': location,
        'description': f'Explore {attraction_name} in {location}',
        'start': {'date': visit_date, 'timeZone': 'Africa/Nairobi'},
        'end': {'date': visit_date, 'timeZone': 'Africa/Nairobi'},
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 60},
            ],
        },
    }

    try:
        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")
        return jsonify({"message": "Event added to your Google Calendar!", "event_link": event.get('htmlLink')})
    except Exception as e:
        print(f"Error creating event: {e}")
        return jsonify({"error": f"Failed to create event: {str(e)}"}), 500
    
# Submit a rating for an attraction
@app.route('/rate_attraction', methods=['POST'])
@login_required
def rate_attraction():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    attraction_name = data.get('attraction_name')
    rating = data.get('rating')
    user_id = get_user_id()  # Reuse your existing function

    if not attraction_name or not rating:
        return jsonify({"error": "Missing attraction_name or rating"}), 400

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
    except ValueError:
        return jsonify({"error": "Invalid rating value. Must be an integer between 1 and 5"}), 400

    connection = create_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO attraction_ratings (user_id, attraction_name, rating)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, attraction_name) 
        DO UPDATE SET rating = EXCLUDED.rating, created_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (user_id, attraction_name, rating))
        connection.commit()
        print(f"Rating {rating} saved for {attraction_name} by user {user_id}")
        
        # Award points for rating (optional gamification)
        add_points(user_id, 5)  # 5 points for submitting a rating
        stats = get_user_stats(user_id)
        
        return jsonify({"message": "Rating submitted successfully", "gamification": stats})
    except Exception as e:
        print(f"Error saving rating: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Get average rating and user's rating for an attraction
@app.route('/get_attraction_ratings', methods=['GET'])
@login_required
def get_attraction_ratings():
    attraction_name = request.args.get('attraction_name')
    user_id = get_user_id()

    if not attraction_name:
        return jsonify({"error": "Missing attraction_name"}), 400

    connection = create_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()
        # Get user's rating
        cursor.execute("""
            SELECT rating FROM attraction_ratings 
            WHERE user_id = %s AND attraction_name = %s
        """, (user_id, attraction_name))
        user_rating = cursor.fetchone()
        user_rating = user_rating[0] if user_rating else None

        # Get average rating
        cursor.execute("""
            SELECT AVG(rating) FROM attraction_ratings 
            WHERE attraction_name = %s
        """, (attraction_name,))
        avg_rating = cursor.fetchone()[0]
        avg_rating = float(avg_rating) if avg_rating else 0.0

        return jsonify({
            "attraction_name": attraction_name,
            "user_rating": user_rating,
            "average_rating": round(avg_rating, 1)  # Rounded to 1 decimal
        })
    except Exception as e:
        print(f"Error fetching ratings: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Routes for templates
@app.route('/')
def root():
    if 'user' in session or 'email' in session:
        return redirect(url_for('index'))
    return redirect(url_for('signup_page'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')

@app.route('/cookie_policy')
@login_required
def cookie_policy():
    return render_template('cookie_policy.html')

@app.route('/faqs')
@login_required
def faqs():
    return render_template('faqs.html')

@app.route('/privacy_policy')
@login_required
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms_of_service')
@login_required
def terms_of_service():
    return render_template('terms_of_service.html')

@app.route('/tourist_filter')
@login_required
def tourist_filter():
    api_key = os.getenv('OPENWEATHER_API_KEY', 'default_key_if_not_set')
    return render_template('tourist_filter.html', openweather_api_key=api_key)

@app.route('/visitor_info')
@login_required
def visitor_info():
    return render_template('visitor_info.html')

@app.route('/what_to_do')
@login_required
def what_to_do():
    return render_template('what_to_do.html')

@app.route('/where_to_eat')
@login_required
def where_to_eat():
    return render_template('where_to_eat.html')

@app.route('/where_to_stay')
@login_required
def where_to_stay():
    return render_template('where_to_stay.html')

@app.route('/get_weather')
@login_required
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return jsonify({"error": "API key not found"}), 500
    response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric')
    if response.status_code != 200:
        return jsonify({"error": f"Weather API error: {response.status_code} - {response.text}"}), response.status_code
    return jsonify(response.json())

@app.route('/filter_attractions', methods=['POST'])
@login_required
def filter_attractions():
    filters = request.get_json()
    user_id = get_user_id()
    add_points(user_id, 10)  # Award 10 points for filtering

    location = filters.get('location')
    budget = filters.get('budget')
    experience = filters.get('experience')

    print(f"Received filters: {filters}")

    budget_min = None
    budget_max = None
    if budget:
        if budget == "10000+":
            budget_min = 10000
            budget_max = 100000
        else:
            budget_range = budget.split('-')
            budget_min = int(budget_range[0])
            budget_max = int(budget_range[1])

    connection = create_connection()
    if not connection:
        print("Database connection failed")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor()
        query = """
        SELECT name, location, description, activities, best_time_to_visit, 
               rates_citizens, rates_residents, rates_non_residents, latitude, longitude
        FROM attractions
        WHERE (%s IS NULL OR location ILIKE %s)
        """
        params = [location, f"%{location}%" if location else None]

        print(f"Executing base query with params: {params}")
        cursor.execute(query, params)
        results = cursor.fetchall()
        print(f"Raw database results: {results}")

        attractions = []
        for result in results:
            name, loc, desc, activities, best_time, citizens, residents, non_res, lat, lon = result
            rate_value = int(citizens.replace('Ksh ', '').replace(',', '')) if citizens != 'N/A' else 0

            if budget_min is not None and (rate_value < budget_min or rate_value > budget_max):
                continue
            if experience and experience not in activities:
                continue

            attractions.append({
                "name": name,
                "location": loc,
                "description": desc,
                "experience": activities,
                "best_time_to_visit": best_time,
                "rates_citizens": citizens,
                "rates_residents": residents,
                "rates_non_residents": non_res,
                "lat": lat if lat is not None else 0,
                "lon": lon if lon is not None else 0
            })

        stats = get_user_stats(user_id)
        print(f"Final filtered results: {len(attractions)} attractions: {attractions}")
        return jsonify({"attractions": attractions, "gamification": stats})

    except Exception as e:
        print(f"Error querying database: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_id = get_user_id()
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        response = ask_question(query)
        add_points(user_id, 20)
        stats = get_user_stats(user_id)
        return jsonify({"response": response, "gamification": stats})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/gamification_stats', methods=['GET'])
@login_required
def gamification_stats():
    user_id = get_user_id()
    stats = get_user_stats(user_id)
    return jsonify(stats)

@app.route('/search_hotels', methods=['POST'])
@login_required
def search_hotels():
    # Get the request data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided in the request"}), 400

    # Extract and validate input data
    check_in = data.get('checkIn')  # e.g., "2025-04-01"
    check_out = data.get('checkOut')  # e.g., "2025-04-01"
    location = data.get('location', 'KE')  # Default to Kenya
    adults = data.get('adults', 1)
    user_id = get_user_id()

    # Validate required fields
    if not check_in or not check_out:
        return jsonify({"error": "Check-in and check-out dates are required"}), 400

    # Validate date format and ensure checkOut is after checkIn
    try:
        check_in_date = datetime.strptime(check_in, '%Y-%m-%d')
        check_out_date = datetime.strptime(check_out, '%Y-%m-%d')
        if check_in_date >= check_out_date:
            return jsonify({"error": "Check-out date must be after check-in date"}), 400
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    try:
        # Validate adults as an integer
        adults = int(adults)
        if adults < 1:
            raise ValueError("Number of adults must be at least 1")
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid number of adults"}), 400

    # Generate signature for Hotelbeds API
    signature, timestamp = generate_signature()

    # Map user-friendly location names to Hotelbeds destination codes
    location_mapping = {
        "KE": "KE",
        "NBO": "NBO",
        "MBA": "MBA",
        "KIS": "KIS",
        "DIA": "DIA",
    }

    # Determine if we're searching by country or specific destination
    destination_code = location_mapping.get(location.upper(), "KE")
    payload = {
        "stay": {
            "checkIn": check_in,
            "checkOut": check_out
        },
        "occupancies": [
            {
                "rooms": 1,
                "adults": adults,
                "children": 0,
                "paxes": []
            }
        ],
        "language": "ENG"
    }

    # Temporarily force a country-wide search for Nairobi to debug
    if destination_code == "NBO":
        print("Forcing country-wide search for Nairobi to debug...")
        # Since destination.countryCode is not working, use a specific destinationCode
        payload["destination"] = {"code": "NBO"}  # Fallback to NBO instead of countryCode
        destination_code = "NBO"
    elif destination_code == "KE":
        # Since destination.countryCode is not working, use a default destinationCode
        payload["destination"] = {"code": "NBO"}  # Fallback to NBO
        destination_code = "NBO"
    else:
        payload["destination"] = {"code": destination_code}

    headers = {
        "Api-Key": HOTELBEDS_API_KEY,
        "X-Signature": signature,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print(f"Sending Hotelbeds API request with payload: {payload}")
    print(f"Request headers: {headers}")

    try:
        response = requests.post(f"{HOTELBEDS_BASE_URL}/hotels", json=payload, headers=headers)
        response.raise_for_status()
        hotels_data = response.json()
        print(f"Hotelbeds API response for {destination_code}: {hotels_data}")

        # Validate the response structure
        if not isinstance(hotels_data, dict):
            raise ValueError("Unexpected API response format: Response is not a dictionary")

        hotels_list = hotels_data.get("hotels", {}).get("hotels", [])
        if not isinstance(hotels_list, list):
            raise ValueError("Unexpected API response format: 'hotels.hotels' is not a list")

        print(f"Total hotels returned by API before filtering: {len(hotels_list)}")
        if len(hotels_list) == 0:
            print(f"No hotels found for destination {destination_code}")

        # Extract and filter hotels
        hotels = []
        for hotel in hotels_list:
            if not isinstance(hotel, dict):
                print(f"Skipping invalid hotel entry: {hotel}")
                continue

            # Log the full hotel object for debugging
            print(f"Processing hotel: {hotel}")

            # Check countryCode (secondary filter)
            country_code = hotel.get("countryCode", "")
            if country_code and country_code != "KE":
                print(f"Excluding hotel {hotel.get('name', 'Unknown')} - Country code ({country_code}) is not KE")
                continue

            # Extract latitude and longitude with error handling
            try:
                lat = float(hotel.get("latitude", 0))
                lon = float(hotel.get("longitude", 0))
            except (ValueError, TypeError) as e:
                print(f"Excluding hotel {hotel.get('name', 'Unknown')} - Invalid coordinates: {e}")
                continue

            # Kenya's geographical boundaries (primary filter)
            if not (lat >= -4.5 and lat <= 5.0 and lon >= 33.9 and lon <= 41.9):
                print(f"Excluding hotel {hotel.get('name', 'Unknown')} - Coordinates ({lat}, {lon}) are outside Kenya")
                continue

            # If searching for a specific destination, ensure the destinationCode matches
            hotel_destination_code = hotel.get("destinationCode", "")
            if destination_code != "KE" and hotel_destination_code != destination_code:
                print(f"Excluding hotel {hotel.get('name', 'Unknown')} - Destination code ({hotel_destination_code}) does not match requested ({destination_code})")
                continue

            # For Nairobi search, ensure the hotel is in Nairobi by destinationCode or coordinates
            if location.upper() == "NBO":
                if hotel_destination_code != "NBO":
                    # Nairobi coordinates are roughly lat: -1.2864, lon: 36.8172
                    # Allow a small range around Nairobi (e.g., ±1 degree)
                    if not (lat >= -2.3 and lat <= -0.3 and lon >= 35.8 and lon <= 37.8):
                        print(f"Excluding hotel {hotel.get('name', 'Unknown')} - Not in Nairobi (destinationCode: {hotel_destination_code}, coordinates: {lat}, {lon})")
                        continue

            # Safely extract the hotel name
            name = "Unknown Hotel"
            if isinstance(hotel.get("name"), dict):
                name = hotel["name"].get("content", "Unknown Hotel")
            elif isinstance(hotel.get("name"), str):
                name = hotel["name"]

            hotels.append({
                "name": name,
                "minRate": hotel.get("minRate", "0"),
                "currency": hotel.get("currency", "Unknown"),
                "latitude": lat,
                "longitude": lon,
                "destinationName": hotel.get("destinationName", "Unknown"),
                "destinationCode": hotel_destination_code,
                "countryCode": country_code if country_code else "Unknown"
            })

        # Award points for searching hotels
        add_points(user_id, 15)  # 15 points for hotel search
        stats = get_user_stats(user_id)

        return jsonify({
            "hotels": hotels,
            "checkIn": check_in,
            "checkOut": check_out,
            "total": len(hotels),
            "gamification": stats
        })

    except requests.RequestException as e:
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Hotelbeds API error response: {error_detail}")
            except ValueError:
                print(f"Hotelbeds API error response (non-JSON): {e.response.text}")
        else:
            print(f"Error fetching hotels: {e}")
        return jsonify({"error": str(e)}), 500

# Redirect routes for incorrect links in index.html
@app.route('/tourist_filter.html')
def tourist_filter_html_redirect():
    return redirect(url_for('tourist_filter'))

@app.route('/what_to_do.html')
def what_to_do_html_redirect():
    return redirect(url_for('what_to_do'))

@app.route('/where_to_stay.html')
def where_to_stay_html_redirect():
    return redirect(url_for('where_to_stay'))

@app.route('/where_to_eat.html')
def where_to_eat_html_redirect():
    return redirect(url_for('where_to_eat'))

@app.route('/visitor_info.html')
def visitor_info_html_redirect():
    return redirect(url_for('visitor_info'))

@app.route('/contact.html')
def contact_html_redirect():
    return redirect(url_for('contact'))
    
if __name__ == '__main__':
    app.run(debug=True)