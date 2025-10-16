"""
Login Module for SynchroChain AI Dashboard
Handles user authentication and role-based access
"""
import streamlit as st
import hashlib
import json
import os
from datetime import datetime

def hash_password(password):
    """Simple password hashing for demo purposes."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_credentials():
    """Load user credentials from file or return defaults."""
    credentials_file = "data/user_credentials.json"
    
    if os.path.exists(credentials_file):
        try:
            with open(credentials_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default credentials for demo
    default_credentials = {
        "admin": {
            "password": hash_password("admin123"),
            "role": "admin",
            "name": "Admin User",
            "email": "admin@synchrochain.com"
        },
        "user": {
            "password": hash_password("user123"),
            "role": "user", 
            "name": "Demo User",
            "email": "user@synchrochain.com"
        },
        "analyst": {
            "password": hash_password("analyst123"),
            "role": "analyst",
            "name": "Data Analyst",
            "email": "analyst@synchrochain.com"
        }
    }
    
    # Save default credentials
    os.makedirs("data", exist_ok=True)
    with open(credentials_file, 'w') as f:
        json.dump(default_credentials, f, indent=2)
    
    return default_credentials

def display_login_page():
    """Display the login page."""
    st.markdown("### 🔐 Authentication Required")
    st.markdown("Please log in to access the SynchroChain AI Dashboard")
    
    # Create two columns for login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("#### Login Form")
            
            with st.form("login_form"):
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    help="Use 'admin', 'user', or 'analyst' for demo"
                )
                
                password = st.text_input(
                    "Password", 
                    type="password",
                    placeholder="Enter your password",
                    help="Use 'admin123', 'user123', or 'analyst123' for demo"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    login_button = st.form_submit_button(
                        "🚀 Login",
                        use_container_width=True
                    )
                
                with col_btn2:
                    if st.form_submit_button("🔄 Reset", use_container_width=True):
                        st.rerun()
            
            # Handle login
            if login_button:
                if authenticate_user(username, password):
                    st.success(f"✅ Welcome, {username.title()}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            
            # Demo credentials info
            st.markdown("---")
            st.markdown("#### 🎯 Demo Credentials")
            
            credentials_info = {
                "Admin": "admin / admin123",
                "User": "user / user123", 
                "Analyst": "analyst / analyst123"
            }
            
            for role, creds in credentials_info.items():
                st.code(f"{role}: {creds}")

def authenticate_user(username, password):
    """Authenticate user with credentials."""
    if not username or not password:
        return False
    
    credentials = load_user_credentials()
    
    if username in credentials:
        stored_password = credentials[username]["password"]
        hashed_input = hash_password(password)
        
        if stored_password == hashed_input:
            # Set session state
            st.session_state.logged_in = True
            st.session_state.role = credentials[username]["role"]
            st.session_state.username = username
            st.session_state.user_info = {
                "name": credentials[username]["name"],
                "email": credentials[username]["email"]
            }
            
            # Debug: Print the role being set
            print(f"DEBUG: User {username} logging in with role: {credentials[username]['role']}")
            
            # Log login event
            log_login_event(username, credentials[username]["role"])
            
            return True
    
    return False

def log_login_event(username, role):
    """Log login events for audit purposes."""
    log_file = "data/login_logs.json"
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    
    # Add new log entry
    log_entry = {
        "username": username,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "ip_address": "127.0.0.1"  # Demo IP
    }
    
    logs.append(log_entry)
    
    # Keep only last 100 entries
    if len(logs) > 100:
        logs = logs[-100:]
    
    # Save logs
    os.makedirs("data", exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

def create_new_user(username, password, role, name, email):
    """Create a new user account."""
    credentials_file = "data/user_credentials.json"
    credentials = load_user_credentials()
    
    if username in credentials:
        return False, "Username already exists"
    
    credentials[username] = {
        "password": hash_password(password),
        "role": role,
        "name": name,
        "email": email
    }
    
    with open(credentials_file, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    return True, "User created successfully"

def get_user_stats():
    """Get user statistics for admin dashboard."""
    log_file = "data/login_logs.json"
    
    if not os.path.exists(log_file):
        return {"total_logins": 0, "recent_logins": []}
    
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        # Get recent logins (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_logins = [
            log for log in logs 
            if datetime.fromisoformat(log["timestamp"]) > recent_cutoff
        ]
        
        return {
            "total_logins": len(logs),
            "recent_logins": len(recent_logins),
            "last_login": logs[-1]["timestamp"] if logs else None
        }
    except:
        return {"total_logins": 0, "recent_logins": 0}
