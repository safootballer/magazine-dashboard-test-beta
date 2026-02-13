import streamlit as st
import hashlib
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --------------------------------------------------
# Init
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Database Setup with PostgreSQL (Production)
# --------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    st.error("❌ DATABASE_URL environment variable not set!")
    st.stop()

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=False
)

Base = declarative_base()

# --------------------------------------------------
# Database Models
# --------------------------------------------------
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
    best_players = Column(Text)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(String(255))
    last_login = Column(String(255))

class GenerationCost(Base):
    __tablename__ = 'generation_costs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer)
    match_id = Column(String(255))
    content_type = Column(String(255))
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost_usd = Column(Float)
    model = Column(String(100))
    generated_at = Column(String(255))

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
    page_title="Admin Dashboard - Sports Magazine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.98);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }
    
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem !important;
    }
    
    h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 1rem 0 0.5rem 0;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 1.5rem !important;
    }
    
    h3 {
        color: #1e40af !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(59, 130, 246, 0.3);
        transform: translateX(5px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.4);
    }
    
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stDateInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left-color: #3b82f6;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: #ef4444;
    }
    
    div[data-testid="stArrowVegaLiteChart"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    .caption {
        color: #64748b;
        font-size: 0.875rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .cost-highlight {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        font-weight: 600;
        color: #92400e;
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Administrator Access")
        st.markdown("Please sign in with your admin credentials")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input("👤 Admin Username", placeholder="Enter username", key="admin_username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password", key="admin_password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Sign In", use_container_width=True):
            if username and password:
                with st.spinner("Authenticating..."):
                    admin = verify_admin_login(username, password)
                    if admin:
                        st.session_state.admin_logged_in = True
                        st.session_state.admin = admin
                        st.success(f"✅ Welcome back, {admin['username']}!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials or insufficient permissions")
            else:
                st.warning("⚠️ Please enter both username and password")

        st.markdown('</div>', unsafe_allow_html=True)

def logout():
    st.session_state.admin_logged_in = False
    st.session_state.admin = None
    st.rerun()

# --------------------------------------------------
# Dashboard Pages
# --------------------------------------------------
def show_statistics():
    st.header("📊 System Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        col1, col2, col3, col4 = st.columns(4)

        total_users = db.query(User).count()
        col1.metric("👥 Total Users", total_users)

        total_matches = db.query(Match).count()
        col2.metric("🏈 Total Matches", total_matches)

        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent_matches = db.query(Match).filter(Match.extracted_at >= seven_days_ago).count()
        col3.metric("📅 Matches (7 Days)", recent_matches)

        total_generations = db.query(GenerationCost).count()
        col4.metric("✨ Generations", total_generations)

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        st.subheader("💰 Content Generation Costs")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)

        current_month = datetime.utcnow().strftime('%Y-%m')
        current_month_cost = db.query(func.sum(GenerationCost.cost_usd)).filter(
            func.substr(GenerationCost.generated_at, 1, 7) == current_month
        ).scalar() or 0.0
        col1.metric("💵 This Month", f"${current_month_cost:.4f}")

        total_cost = db.query(func.sum(GenerationCost.cost_usd)).scalar() or 0.0
        col2.metric("💎 All Time", f"${total_cost:.4f}")

        avg_cost = db.query(func.avg(GenerationCost.cost_usd)).scalar() or 0.0
        col3.metric("📊 Avg Cost", f"${avg_cost:.6f}")

        monthly_generations = db.query(GenerationCost).filter(
            func.substr(GenerationCost.generated_at, 1, 7) == current_month
        ).count()
        col4.metric("🔄 Monthly Gen", monthly_generations)

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        st.subheader("🕒 Recent Generations")
        st.markdown("<br>", unsafe_allow_html=True)
        
        recent_gens = db.query(
            GenerationCost.generated_at,
            User.username,
            GenerationCost.content_type,
            GenerationCost.prompt_tokens,
            GenerationCost.completion_tokens,
            GenerationCost.total_tokens,
            GenerationCost.cost_usd,
            GenerationCost.model
        ).outerjoin(User, GenerationCost.user_id == User.id)\
         .order_by(GenerationCost.generated_at.desc())\
         .limit(10).all()

        if recent_gens:
            df = pd.DataFrame(recent_gens, columns=[
                "Date", "User", "Content Type", "Prompt Tokens", 
                "Completion Tokens", "Total Tokens", "Cost (USD)", "Model"
            ])
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M")
            df["Cost (USD)"] = df["Cost (USD)"].apply(lambda x: f"${x:.6f}")
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
        else:
            st.info("📭 No generation history yet")
    finally:
        db.close()

def show_analytics():
    st.header("📈 Analytics & Insights")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        st.subheader("💰 Cost Over Time (Daily)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            func.date(GenerationCost.generated_at).label('day'),
            func.sum(GenerationCost.cost_usd).label('total_cost'),
            func.count(GenerationCost.id).label('generations')
        ).group_by(func.date(GenerationCost.generated_at))\
         .order_by(func.date(GenerationCost.generated_at)).all()

        if results:
            df = pd.DataFrame(results, columns=['day', 'total_cost', 'generations'])
            df["day"] = pd.to_datetime(df["day"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(df.set_index("day")["total_cost"], height=300)
                st.markdown('<p class="caption">📊 Daily Cost ($)</p>', unsafe_allow_html=True)
            with col2:
                st.line_chart(df.set_index("day")["generations"], height=300)
                st.markdown('<p class="caption">🔄 Daily Generations</p>', unsafe_allow_html=True)
        else:
            st.info("📭 No cost data available")

        st.divider()

        st.subheader("📝 Cost by Content Type")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            GenerationCost.content_type,
            func.count(GenerationCost.id).label('generations'),
            func.sum(GenerationCost.cost_usd).label('total_cost'),
            func.avg(GenerationCost.cost_usd).label('avg_cost')
        ).group_by(GenerationCost.content_type)\
         .order_by(func.sum(GenerationCost.cost_usd).desc()).all()

        if results:
            df = pd.DataFrame(results, columns=['content_type', 'generations', 'total_cost', 'avg_cost'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df.set_index("content_type")["total_cost"], height=300)
                st.markdown('<p class="caption">💵 Total Cost by Type ($)</p>', unsafe_allow_html=True)
            with col2:
                st.dataframe(df, use_container_width=True, height=300, hide_index=True)
        else:
            st.info("📭 No content type data available")

        st.divider()

        st.subheader("👤 Cost by User")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            User.username,
            func.count(GenerationCost.id).label('generations'),
            func.sum(GenerationCost.cost_usd).label('total_cost'),
            func.avg(GenerationCost.cost_usd).label('avg_cost')
        ).outerjoin(GenerationCost, User.id == GenerationCost.user_id)\
         .group_by(User.username)\
         .order_by(func.sum(GenerationCost.cost_usd).desc()).all()

        if results:
            df = pd.DataFrame(results, columns=['username', 'generations', 'total_cost', 'avg_cost'])
            st.dataframe(df, use_container_width=True, height=300, hide_index=True)
        else:
            st.info("📭 No user cost data available")

        st.divider()

        st.subheader("🏈 Matches Created Over Time")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            func.date(Match.extracted_at).label('day'),
            func.count(Match.id).label('total')
        ).group_by(func.date(Match.extracted_at))\
         .order_by(func.date(Match.extracted_at)).all()

        if results:
            df = pd.DataFrame(results, columns=['day', 'total'])
            df["day"] = pd.to_datetime(df["day"])
            st.line_chart(df.set_index("day"), height=300)
        else:
            st.info("📭 No match data available")

        st.divider()

        st.subheader("🏆 Matches by Competition")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            Match.competition,
            func.count(Match.id).label('total')
        ).group_by(Match.competition)\
         .order_by(func.count(Match.id).desc()).all()

        if results:
            df = pd.DataFrame(results, columns=['competition', 'total'])
            st.bar_chart(df.set_index("competition"), height=300)
        else:
            st.info("📭 No competition data available")

        st.divider()

        st.subheader("👥 User Roles Distribution")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            User.role,
            func.count(User.id).label('total')
        ).group_by(User.role).all()

        if results:
            df = pd.DataFrame(results, columns=['role', 'total'])
            st.bar_chart(df.set_index("role"), height=300)
        else:
            st.info("📭 No user data available")
    finally:
        db.close()

def show_cost_management():
    st.header("💰 Cost Management")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        st.subheader("📅 Monthly Cost Breakdown")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            func.substr(GenerationCost.generated_at, 1, 7).label('month'),
            func.count(GenerationCost.id).label('generations'),
            func.sum(GenerationCost.cost_usd).label('total_cost'),
            func.avg(GenerationCost.cost_usd).label('avg_cost'),
            func.sum(GenerationCost.prompt_tokens).label('total_prompt_tokens'),
            func.sum(GenerationCost.completion_tokens).label('total_completion_tokens'),
            func.sum(GenerationCost.total_tokens).label('total_tokens')
        ).group_by(func.substr(GenerationCost.generated_at, 1, 7))\
         .order_by(func.substr(GenerationCost.generated_at, 1, 7).desc()).all()

        if results:
            df = pd.DataFrame(results, columns=[
                'month', 'generations', 'total_cost', 'avg_cost',
                'total_prompt_tokens', 'total_completion_tokens', 'total_tokens'
            ])
            df["total_cost"] = df["total_cost"].apply(lambda x: f"${x:.4f}")
            df["avg_cost"] = df["avg_cost"].apply(lambda x: f"${x:.6f}")
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
        else:
            st.info("📭 No cost data available")

        st.divider()

        st.subheader("🤖 Model Usage & Costs")
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = db.query(
            GenerationCost.model,
            func.count(GenerationCost.id).label('generations'),
            func.sum(GenerationCost.cost_usd).label('total_cost'),
            func.avg(GenerationCost.cost_usd).label('avg_cost'),
            func.sum(GenerationCost.total_tokens).label('total_tokens')
        ).group_by(GenerationCost.model)\
         .order_by(func.sum(GenerationCost.cost_usd).desc()).all()

        if results:
            df = pd.DataFrame(results, columns=['model', 'generations', 'total_cost', 'avg_cost', 'total_tokens'])
            df["total_cost"] = df["total_cost"].apply(lambda x: f"${x:.6f}")
            df["avg_cost"] = df["avg_cost"].apply(lambda x: f"${x:.6f}")
            st.dataframe(df, use_container_width=True, height=300, hide_index=True)
        else:
            st.info("📭 No model usage data available")

        st.divider()

        st.subheader("📥 Export Cost Data")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("📅 Start Date", value=pd.Timestamp.now().date().replace(day=1))
        with col2:
            end_date = st.date_input("📅 End Date", value=pd.Timestamp.now().date())

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📥 Export to CSV", use_container_width=True):
            with st.spinner("Generating export..."):
                results = db.query(
                    GenerationCost.generated_at,
                    User.username,
                    GenerationCost.content_type,
                    GenerationCost.prompt_tokens,
                    GenerationCost.completion_tokens,
                    GenerationCost.total_tokens,
                    GenerationCost.cost_usd,
                    GenerationCost.model,
                    GenerationCost.match_id
                ).outerjoin(User, GenerationCost.user_id == User.id)\
                 .filter(func.date(GenerationCost.generated_at).between(start_date, end_date))\
                 .order_by(GenerationCost.generated_at.desc()).all()
                
                if results:
                    df = pd.DataFrame(results, columns=[
                        'generated_at', 'username', 'content_type', 'prompt_tokens',
                        'completion_tokens', 'total_tokens', 'cost_usd', 'model', 'match_id'
                    ])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv,
                        file_name=f"generation_costs_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success(f"✅ Ready to download! {len(df)} records found.")
                else:
                    st.warning("⚠️ No data available for the selected date range")
    finally:
        db.close()

def manage_users():
    st.header("👥 User Management")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("➕ Add New User"):
        st.markdown("### Create New User Account")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("👤 Username", placeholder="Enter username")
            role = st.selectbox("🎭 Role", ["user", "admin"])
        
        with col2:
            password = st.text_input("🔒 Password", type="password", placeholder="Enter secure password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✨ Create User", use_container_width=True):
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
                        st.success("✅ User created successfully!")
                        st.balloons()
                        st.rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Error: {e}")
                finally:
                    db.close()
            else:
                st.error("⚠️ Please fill all fields")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📋 All Users")
    st.markdown("<br>", unsafe_allow_html=True)
    
    db = get_db()
    try:
        users = db.query(
            User.id,
            User.username,
            User.role,
            User.created_at,
            User.last_login,
            func.count(GenerationCost.id).label('generations'),
            func.sum(GenerationCost.cost_usd).label('total_cost')
        ).outerjoin(GenerationCost, User.id == GenerationCost.user_id)\
         .group_by(User.id)\
         .order_by(User.created_at.desc()).all()

        for u in users:
            with st.expander(f"👤 {u[1]} · {u[2].upper()}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📅 Account Info**")
                    st.write(f"Created: {u[3][:10] if u[3] else 'N/A'}")
                    st.write(f"Last Login: {u[4][:10] if u[4] else 'Never'}")
                
                with col2:
                    st.markdown("**📊 Usage Stats**")
                    st.write(f"Generations: {u[5] or 0}")
                    st.markdown(f'<div class="cost-highlight">Total Cost: ${(u[6] or 0):.6f}</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**⚙️ Actions**")
                    if u[1] != "admin":
                        if st.button("🗑️ Delete User", key=f"del_{u[0]}", use_container_width=True):
                            db_del = get_db()
                            try:
                                user_to_delete = db_del.query(User).filter_by(id=u[0]).first()
                                if user_to_delete:
                                    db_del.query(GenerationCost).filter_by(user_id=u[0]).delete()
                                    db_del.delete(user_to_delete)
                                    db_del.commit()
                                    st.success("✅ User deleted")
                                    st.rerun()
                            finally:
                                db_del.close()
                    else:
                        st.info("🔒 Protected Account")
    finally:
        db.close()

def manage_matches():
    st.header("🏈 Match Management")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        total_matches = db.query(Match).count()
        st.info(f"📊 Total Matches: {total_matches}")
        st.markdown("<br>", unsafe_allow_html=True)

        matches_per_page = 20
        total_pages = (total_matches + matches_per_page - 1) // matches_per_page
        
        if total_pages > 1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            offset = (page - 1) * matches_per_page
        else:
            offset = 0

        matches = db.query(Match)\
            .order_by(Match.date.desc())\
            .limit(matches_per_page)\
            .offset(offset)\
            .all()

        for m in matches:
            with st.expander(f"🏈 {m.home_team} vs {m.away_team} · {m.date}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Match Details**")
                    st.write(f"🏆 Competition: {m.competition}")
                    st.write(f"📍 Venue: {m.venue}")
                    st.write(f"📊 Score: **{m.home_team} {m.home_final_score}** – **{m.away_team} {m.away_final_score}**")
                    st.write(f"📏 Margin: **{m.margin}** points")
                
                with col2:
                    st.markdown("**⚽ Goal Scorers**")
                    if m.goal_scorers:
                        scorers = json.loads(m.goal_scorers)
                        for team, team_scorers in scorers.items():
                            st.write(f"**{team}:**")
                            if team_scorers:
                                for scorer in team_scorers:
                                    st.write(f"  • {scorer}")
                            else:
                                st.write("  None")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if m.best_players:
                        st.markdown("**⭐ Best Players**")
                        best_players = json.loads(m.best_players)
                        for team, players in best_players.items():
                            st.write(f"**{team}:**")
                            if players:
                                for player in players:
                                    st.write(f"  • {player}")
                            else:
                                st.write("  Not available")

                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🗑️ Delete Match", key=f"match_{m.id}", use_container_width=True):
                    db_del = get_db()
                    try:
                        match_to_delete = db_del.query(Match).filter_by(id=m.id).first()
                        if match_to_delete:
                            db_del.delete(match_to_delete)
                            db_del.commit()
                            st.success("✅ Match deleted")
                            st.rerun()
                    finally:
                        db_del.close()
    finally:
        db.close()

def admin_dashboard():
    st.title("⚙️ Admin Dashboard")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<span class="badge">👤 {st.session_state.admin["username"]} · {st.session_state.admin["role"].upper()}</span>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    st.divider()

    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "📈 Analytics", "💰 Cost Management", "👥 Users", "🏈 Matches"],
        label_visibility="collapsed"
    )

    if page == "📊 Dashboard":
        show_statistics()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "💰 Cost Management":
        show_cost_management()
    elif page == "👥 Users":
        manage_users()
    elif page == "🏈 Matches":
        manage_matches()

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if st.session_state.admin_logged_in:
    admin_dashboard()
else:
    login_page()
