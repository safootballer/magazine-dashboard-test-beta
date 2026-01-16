import streamlit as st
import sqlite3
import hashlib
import json
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# Database Connection
# --------------------------------------------------
conn = sqlite3.connect("magazine.db", check_same_thread=False)
cur = conn.cursor()

# --------------------------------------------------
# Auth Functions
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin_login(username, password):
    password_hash = hash_password(password)
    result = cur.execute("""
        SELECT id, username, role FROM users 
        WHERE username = ? AND password_hash = ? AND role = 'admin'
    """, (username, password_hash)).fetchone()

    if result:
        cur.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.utcnow().isoformat(), result[0]))
        conn.commit()
        return {"id": result[0], "username": result[1], "role": result[2]}
    return None

# --------------------------------------------------
# Login Page
# --------------------------------------------------
def login_page():
    st.title("⚙️ Admin Dashboard Login")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Administrator Access Only")
        username = st.text_input("Admin Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            admin = verify_admin_login(username, password)
            if admin:
                st.session_state.admin_logged_in = True
                st.session_state.admin = admin
                st.success(f"Welcome, {admin['username']}!")
                st.rerun()
            else:
                st.error("Invalid credentials or insufficient permissions")

        st.info("**Default:** admin / admin123")

def logout():
    st.session_state.admin_logged_in = False
    st.session_state.admin = None
    st.rerun()

# --------------------------------------------------
# Dashboard Pages
# --------------------------------------------------
def show_statistics():
    st.header("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    total_users = cur.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    col1.metric("Total Users", total_users)

    total_matches = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    col2.metric("Total Matches", total_matches)

    recent_matches = cur.execute("""
        SELECT COUNT(*) FROM matches 
        WHERE extracted_at >= datetime('now', '-7 days')
    """).fetchone()[0]
    col3.metric("Matches (Last 7 Days)", recent_matches)

# --------------------------------------------------
# Analytics (GRAPHS)
# --------------------------------------------------
def show_analytics():
    st.header("📈 Analytics & Insights")

    # Matches over time
    st.subheader("🏈 Matches Created Over Time")
    df = pd.read_sql("""
        SELECT date(extracted_at) AS day, COUNT(*) AS total
        FROM matches
        GROUP BY day
        ORDER BY day
    """, conn)

    if not df.empty:
        df["day"] = pd.to_datetime(df["day"])
        st.line_chart(df.set_index("day"))
    else:
        st.info("No match data available")

    st.divider()

    # Matches by competition
    st.subheader("🏆 Matches by Competition")
    df = pd.read_sql("""
        SELECT competition, COUNT(*) AS total
        FROM matches
        GROUP BY competition
        ORDER BY total DESC
    """, conn)

    if not df.empty:
        st.bar_chart(df.set_index("competition"))
    else:
        st.info("No competition data available")

    st.divider()

    # User roles
    st.subheader("👥 User Roles Distribution")
    df = pd.read_sql("""
        SELECT role, COUNT(*) AS total
        FROM users
        GROUP BY role
    """, conn)

    if not df.empty:
        st.bar_chart(df.set_index("role"))
    else:
        st.info("No user data available")

# --------------------------------------------------
# User Management
# --------------------------------------------------
def manage_users():
    st.header("👥 User Management")

    with st.expander("➕ Add New User"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])

        if st.button("Create User"):
            if username and password:
                try:
                    cur.execute("""
                        INSERT INTO users (username, password_hash, role, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (username, hash_password(password), role, datetime.utcnow().isoformat()))
                    conn.commit()
                    st.success("User created successfully")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
            else:
                st.error("Fill all fields")

    st.divider()

    users = cur.execute("""
        SELECT id, username, role, created_at, last_login
        FROM users
        ORDER BY created_at DESC
    """).fetchall()

    for u in users:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
        col1.write(f"**{u[1]}**")
        col2.write(u[2])
        col3.write(u[3][:10] if u[3] else "N/A")
        col4.write(u[4][:10] if u[4] else "Never")

        with col5:
            if u[1] != "admin":
                if st.button("🗑️", key=f"del_{u[0]}"):
                    cur.execute("DELETE FROM users WHERE id=?", (u[0],))
                    conn.commit()
                    st.rerun()

# --------------------------------------------------
# Match Management
# --------------------------------------------------
def manage_matches():
    st.header("🏈 Match Management")

    matches = cur.execute("""
        SELECT id, home_team, away_team, date, competition, venue,
               home_final_score, away_final_score, goal_scorers
        FROM matches
        ORDER BY date DESC
    """).fetchall()

    for m in matches:
        with st.expander(f"{m[1]} vs {m[2]} ({m[3]})"):
            st.write(f"**Competition:** {m[4]}")
            st.write(f"**Venue:** {m[5]}")
            st.write(f"**Score:** {m[1]} {m[6]} – {m[2]} {m[7]}")

            scorers = json.loads(m[8])
            st.write("**Goal Scorers:**")
            st.write(scorers)

            if st.button("🗑️ Delete Match", key=f"match_{m[0]}"):
                cur.execute("DELETE FROM matches WHERE id=?", (m[0],))
                conn.commit()
                st.rerun()

# --------------------------------------------------
# Admin Dashboard
# --------------------------------------------------
def admin_dashboard():
    st.title("⚙️ Admin Dashboard")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"👤 Logged in as **{st.session_state.admin['username']}**")
    with col2:
        if st.button("🚪 Logout"):
            logout()

    st.divider()

    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "📈 Analytics", "👥 Users", "🏈 Matches"]
    )

    if page == "📊 Dashboard":
        show_statistics()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "👥 Users":
        manage_users()
    elif page == "🏈 Matches":
        manage_matches()

# --------------------------------------------------
# Run App
# --------------------------------------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if st.session_state.admin_logged_in:
    admin_dashboard()
else:
    login_page()
