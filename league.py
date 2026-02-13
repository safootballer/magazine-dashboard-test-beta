import streamlit as st
import hashlib
import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --------------------------------------------------
# INIT
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="League Info",
    page_icon="🏈",
    layout="wide"
)

# --------------------------------------------------
# DATABASE SETUP WITH POSTGRESQL
# --------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()

# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------
class Ladder(Base):
    __tablename__ = 'ladder'
    id = Column(Integer, primary_key=True, autoincrement=True)
    grade_id = Column(String(255))
    grade_name = Column(String(255))
    season = Column(String(255))
    round_id = Column(String(255))
    round_name = Column(String(255))
    team_id = Column(String(255))
    team_name = Column(String(255))
    rank = Column(Integer)
    played = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    draws = Column(Integer)
    byes = Column(Integer, default=0)
    points = Column(Integer)
    percentage = Column(Float)
    synced_at = Column(String(255))

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(String(255))
    last_login = Column(String(255))

# Create tables
Base.metadata.create_all(engine)

# Migration function to add missing columns
def migrate_database():
    """Add any missing columns to existing tables"""
    inspector = inspect(engine)
    
    if 'ladder' in inspector.get_table_names():
        existing_columns = [col['name'] for col in inspector.get_columns('ladder')]
        
        if 'byes' not in existing_columns:
            try:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE ladder ADD COLUMN byes INTEGER DEFAULT 0"))
                    conn.commit()
                print("✅ Added byes column to ladder table")
            except Exception as e:
                print(f"⚠️ Could not add byes column: {e}")

migrate_database()

SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stDataFrame {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
        border-radius: 12px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: #ef4444;
        border-radius: 12px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left-color: #3b82f6;
        border-radius: 12px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
        border-radius: 12px;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# AUTH
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    db = get_db()
    try:
        pw_hash = hash_password(password)
        user = db.query(User).filter_by(
            username=username,
            password_hash=pw_hash
        ).first()
        
        if user:
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    finally:
        db.close()

# --------------------------------------------------
# PLAYHQ CONFIG
# --------------------------------------------------
PLAYHQ_URL = "https://api.playhq.com/graphql"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Origin": "https://www.playhq.com",
    "Referer": "https://www.playhq.com/",
    "tenant": "afl"
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_grade_id(url: str):
    if not url:
        return None
    parts = url.rstrip("/").split("/")
    for p in reversed(parts):
        if len(p) >= 6:
            return p
    return None

def safe_post(payload):
    try:
        r = requests.post(
            PLAYHQ_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        if r.status_code != 200:
            st.error(f"PlayHQ request failed ({r.status_code})")
            st.code(r.text)
            return None

        data = r.json()
        if "data" not in data:
            st.error("Invalid PlayHQ response")
            st.code(data)
            return None

        return data["data"]
    except Exception as e:
        st.error(f"Request error: {e}")
        return None

# --------------------------------------------------
# GRAPHQL QUERIES
# --------------------------------------------------
GRADE_META_QUERY = """
query GradeMeta($gradeID: ID!) {
  discoverGrade(gradeID: $gradeID) {
    id
    name
    season { name }
  }
}
"""

LADDER_QUERY = """
query GradeLadder($gradeID: ID!) {
  discoverGrade(gradeID: $gradeID) {
    ladder {
      generatedFrom { id name }
      standings {
        played
        won
        lost
        drawn
        byes
        competitionPoints
        alternatePercentage
        team { id name }
      }
    }
  }
}
"""

# --------------------------------------------------
# DATA FUNCTIONS
# --------------------------------------------------
def fetch_grade_meta(grade_id):
    payload = {
        "query": GRADE_META_QUERY,
        "variables": {"gradeID": grade_id}
    }
    data = safe_post(payload)
    if not data:
        return None

    g = data["discoverGrade"]
    return {
        "id": g["id"],
        "name": g["name"],
        "season": g["season"]["name"]
    }

def sync_ladder(grade_id, grade_name, season):
    synced_at = datetime.utcnow().isoformat()

    payload = {
        "query": LADDER_QUERY,
        "variables": {"gradeID": grade_id}
    }
    data = safe_post(payload)
    if not data:
        return

    grade = data["discoverGrade"]
    if not grade or not grade.get("ladder"):
        st.warning("No ladder data returned")
        return

    db = get_db()
    try:
        for block in grade["ladder"]:
            round_id = block["generatedFrom"]["id"]
            round_name = block["generatedFrom"]["name"]

            for idx, row in enumerate(block["standings"], start=1):
                existing = db.query(Ladder).filter_by(
                    team_id=row["team"]["id"],
                    season=season,
                    round_id=round_id
                ).first()

                if existing:
                    existing.grade_id = grade_id
                    existing.grade_name = grade_name
                    existing.round_name = round_name
                    existing.team_name = row["team"]["name"]
                    existing.rank = idx
                    existing.played = row["played"]
                    existing.wins = row["won"]
                    existing.losses = row["lost"]
                    existing.draws = row["drawn"]
                    existing.byes = row.get("byes", 0)
                    existing.points = row["competitionPoints"]
                    existing.percentage = row["alternatePercentage"]
                    existing.synced_at = synced_at
                else:
                    new_entry = Ladder(
                        grade_id=grade_id,
                        grade_name=grade_name,
                        season=season,
                        round_id=round_id,
                        round_name=round_name,
                        team_id=row["team"]["id"],
                        team_name=row["team"]["name"],
                        rank=idx,
                        played=row["played"],
                        wins=row["won"],
                        losses=row["lost"],
                        draws=row["drawn"],
                        byes=row.get("byes", 0),
                        points=row["competitionPoints"],
                        percentage=row["alternatePercentage"],
                        synced_at=synced_at
                    )
                    db.add(new_entry)

        db.commit()
        st.success("✅ Ladder data synced successfully!")
    except Exception as e:
        db.rollback()
        st.error(f"Database error: {e}")
    finally:
        db.close()

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align: center;'>🏈 League Info Login</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); 
                    padding: 2rem; 
                    border-radius: 16px; 
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);'>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔐 Sign In")
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Login", use_container_width=True, type="primary"):
            if username and password:
                user = verify_login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success(f"✅ Welcome, {user['username']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            else:
                st.warning("⚠️ Please enter both username and password")
        
        st.markdown("</div>", unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main_app():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("# 🏈 League Info & Ladder")
        st.markdown(f"Welcome back, **{st.session_state.user['username']}**! 👋")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    st.divider()

    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {st.session_state.user['username']}")
        st.markdown(f"**Role:** {st.session_state.user['role'].upper()}")
        st.divider()

        st.markdown("### 🔗 PlayHQ League Link")
        league_url = st.text_input(
            "Paste PlayHQ grade URL",
            placeholder="https://www.playhq.com/.../grade/566e0601",
            help="Paste the full URL of a PlayHQ grade/competition page"
        )

        grade_id = extract_grade_id(league_url) if league_url else None
        
        if grade_id:
            st.success(f"✅ Grade ID: `{grade_id}`")
            
            st.divider()
            
            if st.button("🔄 Sync League Data", use_container_width=True, type="primary"):
                with st.spinner("Syncing league ladder from PlayHQ..."):
                    meta = fetch_grade_meta(grade_id)
                    if not meta:
                        st.error("❌ Failed to fetch league metadata")
                        return
                    
                    sync_ladder(meta["id"], meta["name"], meta["season"])
                    st.rerun()
        else:
            if league_url:
                st.error("❌ Invalid PlayHQ URL")

    st.markdown("## 🏆 League Ladder")

    if grade_id:
        db = get_db()
        try:
            results = db.query(
                Ladder.rank,
                Ladder.team_name,
                Ladder.played,
                Ladder.points,
                Ladder.percentage,
                Ladder.wins,
                Ladder.losses,
                Ladder.draws,
                Ladder.byes
            ).filter_by(grade_id=grade_id)\
             .order_by(Ladder.rank).all()

            if results:
                df = pd.DataFrame(results, columns=[
                    "Rank", "Team", "P", "PTS", "%", "W", "L", "D", "BYE"
                ])
                
                meta_info = db.query(Ladder).filter_by(grade_id=grade_id).first()
                if meta_info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🏆 Competition", meta_info.grade_name)
                    col2.metric("📅 Season", meta_info.season)
                    col3.metric("🔄 Last Synced", meta_info.synced_at[:10] if meta_info.synced_at else "N/A")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Team": st.column_config.TextColumn("Team", width="large"),
                        "P": st.column_config.NumberColumn("P", width="small"),
                        "PTS": st.column_config.NumberColumn("PTS", width="small"),
                        "%": st.column_config.NumberColumn("%", format="%.2f", width="small"),
                        "W": st.column_config.NumberColumn("W", width="small"),
                        "L": st.column_config.NumberColumn("L", width="small"),
                        "D": st.column_config.NumberColumn("D", width="small"),
                        "BYE": st.column_config.NumberColumn("BYE", width="small")
                    }
                )
                
                st.caption(f"📊 Showing {len(df)} teams")
            else:
                st.info("📭 No ladder data found. Click **Sync League Data** in the sidebar to load.")
        finally:
            db.close()
    else:
        st.info("🔗 Paste a PlayHQ league link in the sidebar to load ladder data.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📖 How to Use"):
            st.markdown("""
            ### Steps:
            1. Go to PlayHQ website and find your league/grade
            2. Copy the full URL from the browser
            3. Paste it in the sidebar
            4. Click **Sync League Data**
            5. View the ladder table below!
            
            ### Example URL:
```
            https://www.playhq.com/afl/org/adelaide-footy-league/summer-2025/grade/566e0601
```
            """)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
