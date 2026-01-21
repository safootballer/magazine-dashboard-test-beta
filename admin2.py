import streamlit as st
import hashlib
import json
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --------------------------------------------------
# Init
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Database Setup with PostgreSQL
# --------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(255), unique=True)
    extracted_at = Column(String(255))
    date = Column(String(255))
    competition = Column(String(255))
    venue = Column(String(255))
    home_team = Column(String(255))
    away_team = Column(String(255))
    home_final_score = Column(Integer)
    away_final_score = Column(Integer)
    margin = Column(Integer)
    quarter_scores = Column(Text)
    lineups = Column(Text)
    goal_scorers = Column(Text)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(String(255))
    last_login = Column(String(255))

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
}

/* Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    text-align: center;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: #059669;
    margin: 0.5rem 0;
}

.stat-label {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Headers */
h1, h2, h3 {
    color: white !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Tables */
.stDataFrame {
    background: white;
    border-radius: 12px;
    padding: 1rem;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Auth Functions
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin_login(username, password):
    db = get_db()
    try:
        password_hash = hash_password(password)
        user = db.query(User).filter_by(
            username=username, 
            password_hash=password_hash, 
            role='admin'
        ).first()

        if user:
            user.last_login = datetime.utcnow().isoformat()
            db.commit()
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    except Exception as e:
        print(f"Login error: {e}")
        return None
    finally:
        db.close()

# --------------------------------------------------
# Login Page
# --------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align: center; margin-top: 3rem;'>⚙️ Admin Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Administrator Access Only</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 16px; margin-top: 2rem;'>", unsafe_allow_html=True)
        
        username = st.text_input("🔐 Admin Username", placeholder="Enter username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Login to Dashboard", use_container_width=True, type="primary"):
            admin = verify_admin_login(username, password)
            if admin:
                st.session_state.admin_logged_in = True
                st.session_state.admin = admin
                st.success(f"✅ Welcome, {admin['username']}!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Invalid credentials or insufficient permissions")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("ℹ️ Default Credentials"):
            st.code("Username: admin\nPassword: admin123")
        
        st.markdown("</div>", unsafe_allow_html=True)

def logout():
    st.session_state.admin_logged_in = False
    st.session_state.admin = None
    st.rerun()

# --------------------------------------------------
# Dashboard Pages
# --------------------------------------------------
def show_statistics():
    st.markdown("## 📊 System Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        total_users = db.query(User).count()
        total_matches = db.query(Match).count()
        admin_count = db.query(User).filter_by(role='admin').count()
    finally:
        db.close()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">👥</div>
            <div class="stat-number">{total_users}</div>
            <div class="stat-label">Total Users</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">🏈</div>
            <div class="stat-number">{total_matches}</div>
            <div class="stat-label">Total Matches</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem;">⚙️</div>
            <div class="stat-number">{admin_count}</div>
            <div class="stat-label">Administrators</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Recent Activity
    st.markdown("### 📋 Recent Activity")
    
    db = get_db()
    try:
        recent_matches = db.query(Match).order_by(Match.extracted_at.desc()).limit(5).all()
        
        if recent_matches:
            for match in recent_matches:
                with st.expander(f"🏈 {match.home_team} vs {match.away_team} - {match.date}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Competition:** {match.competition}")
                        st.write(f"**Venue:** {match.venue}")
                    with col2:
                        st.write(f"**Final Score:** {match.home_final_score} - {match.away_final_score}")
                        st.write(f"**Margin:** {match.margin} points")
        else:
            st.info("No matches recorded yet")
    finally:
        db.close()

# --------------------------------------------------
# Analytics
# --------------------------------------------------
def show_analytics():
    st.markdown("## 📈 Analytics & Insights")

    db = get_db()
    
    try:
        # Matches by competition
        st.markdown("### 🏆 Matches by Competition")
        
        results = db.query(
            Match.competition, 
            func.count(Match.id).label('total')
        ).group_by(Match.competition).all()
        
        if results:
            df = pd.DataFrame(results, columns=['competition', 'total'])
            st.bar_chart(df.set_index('competition'))
        else:
            st.info("No competition data available")

        st.divider()

        # User roles distribution
        st.markdown("### 👥 User Roles Distribution")
        
        user_results = db.query(
            User.role, 
            func.count(User.id).label('total')
        ).group_by(User.role).all()
        
        if user_results:
            df_users = pd.DataFrame(user_results, columns=['role', 'total'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df_users.set_index('role'))
            with col2:
                for role, count in user_results:
                    st.metric(f"{role.upper()}", count)
        else:
            st.info("No user data available")

        st.divider()

        # Top venues
        st.markdown("### 🏟️ Top Venues")
        
        venue_results = db.query(
            Match.venue, 
            func.count(Match.id).label('total')
        ).group_by(Match.venue).order_by(func.count(Match.id).desc()).limit(10).all()
        
        if venue_results:
            df_venues = pd.DataFrame(venue_results, columns=['venue', 'total'])
            st.dataframe(df_venues, use_container_width=True, hide_index=True)
        else:
            st.info("No venue data available")

    finally:
        db.close()

# --------------------------------------------------
# User Management
# --------------------------------------------------
def manage_users():
    st.markdown("## 👥 User Management")

    with st.expander("➕ Add New User", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username")
            role = st.selectbox("Role", ["user", "admin"])
        
        with col2:
            password = st.text_input("Password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✅ Create User", type="primary"):
            if username and password:
                db = get_db()
                try:
                    existing = db.query(User).filter_by(username=username).first()
                    if existing:
                        st.error("❌ Username already exists")
                    else:
                        new_user = User(
                            username=username,
                            password_hash=hash_password(password),
                            role=role,
                            created_at=datetime.utcnow().isoformat()
                        )
                        db.add(new_user)
                        db.commit()
                        st.success(f"✅ User '{username}' created successfully!")
                        st.rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Error: {e}")
                finally:
                    db.close()
            else:
                st.error("⚠️ Please fill all fields")

    st.divider()
    st.markdown("### 📋 All Users")

    db = get_db()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()

        if users:
            # Header
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
            col1.markdown("**👤 Username**")
            col2.markdown("**🎭 Role**")
            col3.markdown("**📅 Created**")
            col4.markdown("**🕐 Last Login**")
            col5.markdown("**⚙️ Action**")
            
            st.divider()

            # Users list
            for user in users:
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
                
                col1.write(f"**{user.username}**")
                
                if user.role == 'admin':
                    col2.markdown("🔴 **Admin**")
                else:
                    col2.write("🟢 User")
                
                col3.write(user.created_at[:10] if user.created_at else "N/A")
                col4.write(user.last_login[:10] if user.last_login else "Never")

                with col5:
                    if user.username != "admin":
                        if st.button("🗑️", key=f"del_{user.id}"):
                            db_del = get_db()
                            try:
                                user_to_delete = db_del.query(User).filter_by(id=user.id).first()
                                if user_to_delete:
                                    db_del.delete(user_to_delete)
                                    db_del.commit()
                                    st.success("User deleted!")
                                    st.rerun()
                            finally:
                                db_del.close()
                    else:
                        col5.write("🔒")
        else:
            st.info("No users found")
    finally:
        db.close()

# --------------------------------------------------
# Match Management
# --------------------------------------------------
def manage_matches():
    st.markdown("## 🏈 Match Management")

    db = get_db()
    try:
        total_matches = db.query(Match).count()
        st.markdown(f"**Total Matches:** {total_matches}")
        
        st.divider()

        matches = db.query(Match).order_by(Match.date.desc()).all()

        if matches:
            for match in matches:
                with st.expander(f"🏈 {match.home_team} vs {match.away_team} ({match.date})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Competition:** {match.competition}")
                        st.write(f"**Venue:** {match.venue}")
                        st.write(f"**Date:** {match.date}")
                    
                    with col2:
                        st.write(f"**Final Score:** {match.home_team} {match.home_final_score} – {match.away_team} {match.away_final_score}")
                        st.write(f"**Margin:** {match.margin} points")

                    st.divider()

                    # Quarter scores
                    if match.quarter_scores:
                        quarter_data = json.loads(match.quarter_scores)
                        st.write("**Quarter Scores:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{match.home_team}:**")
                            for q, score in quarter_data.get('home', {}).items():
                                st.write(f"{q}: {score}")
                        with col2:
                            st.write(f"**{match.away_team}:**")
                            for q, score in quarter_data.get('away', {}).items():
                                st.write(f"{q}: {score}")

                    # Goal scorers
                    if match.goal_scorers:
                        scorers = json.loads(match.goal_scorers)
                        st.write("**Goal Scorers:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{match.home_team}:**")
                            st.write(", ".join(scorers.get('home', [])) if scorers.get('home') else "None")
                        with col2:
                            st.write(f"**{match.away_team}:**")
                            st.write(", ".join(scorers.get('away', [])) if scorers.get('away') else "None")

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if st.button("🗑️ Delete Match", key=f"match_{match.id}", type="secondary"):
                        db_del = get_db()
                        try:
                            match_to_delete = db_del.query(Match).filter_by(id=match.id).first()
                            if match_to_delete:
                                db_del.delete(match_to_delete)
                                db_del.commit()
                                st.success("Match deleted!")
                                st.rerun()
                        finally:
                            db_del.close()
        else:
            st.info("No matches found")
    finally:
        db.close()

# --------------------------------------------------
# Admin Dashboard
# --------------------------------------------------
def admin_dashboard():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("# ⚙️ Admin Dashboard")
        st.markdown(f"Welcome back, **{st.session_state.admin['username']}** 👋")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    st.divider()

    # Sidebar Navigation
    st.sidebar.markdown("## 📋 Navigation")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "📈 Analytics", "👥 Users", "🏈 Matches"],
        label_visibility="collapsed"
    )

    # Page routing
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
