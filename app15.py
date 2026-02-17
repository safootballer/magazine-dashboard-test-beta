import streamlit as st
import requests
import json
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

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
# Database Models - ALL TABLES
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

class MatchLink(Base):
    __tablename__ = 'match_links'
    id = Column(Integer, primary_key=True, autoincrement=True)
    playhq_url = Column(String(500))
    match_id = Column(String(255), unique=True)
    home_team = Column(String(255))
    away_team = Column(String(255))
    competition = Column(String(255))
    date = Column(String(255))
    venue = Column(String(255))
    added_by = Column(String(255))
    added_at = Column(String(255))
    is_active = Column(Integer, default=1)

# --------------------------------------------------
# AUTO MIGRATION - Creates tables & adds missing columns
# --------------------------------------------------
def run_migrations():
    """Create all tables and patch any missing columns"""
    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        with engine.connect() as conn:
            if 'matches' in existing_tables:
                existing_cols = [c['name'] for c in inspector.get_columns('matches')]
                patches = {
                    'best_players':  'ALTER TABLE matches ADD COLUMN best_players TEXT',
                    'margin':        'ALTER TABLE matches ADD COLUMN margin INTEGER',
                    'lineups':       'ALTER TABLE matches ADD COLUMN lineups TEXT',
                    'goal_scorers':  'ALTER TABLE matches ADD COLUMN goal_scorers TEXT',
                    'quarter_scores':'ALTER TABLE matches ADD COLUMN quarter_scores TEXT',
                }
                for col, sql in patches.items():
                    if col not in existing_cols:
                        try:
                            conn.execute(text(sql))
                            conn.commit()
                        except Exception:
                            pass

            if 'users' in existing_tables:
                existing_cols = [c['name'] for c in inspector.get_columns('users')]
                patches = {
                    'last_login': 'ALTER TABLE users ADD COLUMN last_login VARCHAR(255)',
                    'created_at': 'ALTER TABLE users ADD COLUMN created_at VARCHAR(255)',
                }
                for col, sql in patches.items():
                    if col not in existing_cols:
                        try:
                            conn.execute(text(sql))
                            conn.commit()
                        except Exception:
                            pass

            if 'generation_costs' in existing_tables:
                existing_cols = [c['name'] for c in inspector.get_columns('generation_costs')]
                if 'model' not in existing_cols:
                    try:
                        conn.execute(text('ALTER TABLE generation_costs ADD COLUMN model VARCHAR(100)'))
                        conn.commit()
                    except Exception:
                        pass

        print("✅ DB migration complete")
    except Exception as e:
        print(f"Migration error: {e}")

run_migrations()

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
    page_title="Sports Magazine Automation",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header { text-align: center; color: white; padding: 2rem 0; }
    .login-card {
        background: white; padding: 2rem; border-radius: 15px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .login-card h3, .login-card p, .login-card label { color: #1e293b !important; }
    .feature-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .content-card {
        background: white; padding: 2rem; border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin: 1rem 0;
    }
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 2px solid #cbd5e1; border-radius: 10px; padding: 0.75rem;
    }
    .stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed; box-shadow: 0 0 0 3px rgba(124,58,237,0.1);
        background-color: #ffffff !important; color: #1e293b !important;
    }
    .stTextInput > label { color: #1e293b !important; font-weight: 500; }
    .stSelectbox > div > div {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 2px solid #cbd5e1; border-radius: 10px;
    }
    .stSelectbox > label { color: #1e293b !important; font-weight: 500; }
    .stTextArea textarea {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 2px solid #cbd5e1; border-radius: 10px;
    }
    .stTextArea > label { color: #1e293b !important; font-weight: 500; }
    .stMultiSelect > div > div {
        background-color: #ffffff !important; color: #1e293b !important;
        border: 2px solid #cbd5e1; border-radius: 10px;
    }
    .stMultiSelect > label { color: #1e293b !important; font-weight: 500; }
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span { color: #ffffff !important; }
    /* ===== ALERTS ===== */
    .stSuccess { color: #065f46 !important; }
    .stInfo    { color: #1e3a8a !important; }
    .stWarning { color: #78350f !important; }
    .stError   { color: #7f1d1d !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Logo
# --------------------------------------------------
def render_logo_center():
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("assets/logo.png", width=170)
        except:
            st.markdown("### 📰 Magazine Automation")
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

render_logo_center()

# --------------------------------------------------
# Default Admin
# --------------------------------------------------
def create_default_admin():
    db = get_db()
    try:
        existing_admin = db.query(User).filter_by(username="admin").first()
        if not existing_admin:
            admin_user = User(
                username="admin",
                password_hash=hashlib.sha256("admin123".encode()).hexdigest(),
                role="admin",
                created_at=datetime.utcnow().isoformat()
            )
            db.add(admin_user)
            db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

create_default_admin()

# --------------------------------------------------
# Auth
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    db = get_db()
    try:
        password_hash = hash_password(password)
        user = db.query(User).filter_by(username=username, password_hash=password_hash).first()
        if user:
            user.last_login = datetime.utcnow().isoformat()
            db.commit()
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    except:
        return None
    finally:
        db.close()

# --------------------------------------------------
# Login Page
# --------------------------------------------------
def login_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📰")
    st.markdown('<h1 style="margin: 0; font-size: 3rem; color: white;">Sports Magazine Automation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; margin-top: 0.5rem; color: white;">AI-Powered Match Report Generation</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome Back")
        st.markdown("Sign in to generate professional match reports")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🚀 Sign In", use_container_width=True):
            if username and password:
                user = verify_login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success(f"✅ Welcome back, {user['username']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("⚠️ Please enter both username and password")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="feature-card"><h2>⚡</h2><h4>Fast</h4><p>Generate reports in seconds</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card"><h2>🎯</h2><h4>Accurate</h4><p>Powered by AI technology</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="feature-card"><h2>✍️</h2><h4>Professional</h4><p>Magazine-quality content</p></div>""", unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    if "vectordb" in st.session_state:
        del st.session_state.vectordb
    st.rerun()

# --------------------------------------------------
# PlayHQ API
# --------------------------------------------------
PLAYHQ_GRAPHQL_URL = "https://api.playhq.com/graphql"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Origin": "https://www.playhq.com",
    "Referer": "https://www.playhq.com/",
    "tenant": "afl",
}

GRAPHQL_QUERY = """
query gameView($gameId: ID!) {
  discoverGame(gameID: $gameId) {
    id
    date
    home { ... on DiscoverTeam { name } }
    away { ... on DiscoverTeam { name } }
    allocation { court { venue { name } } }
    round { grade { season { competition { name } } } }
    result { home { score } away { score } }
    statistics {
      home {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant { profile { firstName lastName } }
            ... on DiscoverParticipantFillInPlayer { profile { firstName lastName } }
            ... on DiscoverGamePermitFillInPlayer { profile { firstName lastName } }
          }
          statistics { type { value } count }
        }
        periods { period { value } statistics { type { value } count } }
        bestPlayers { ranking participant { ... on DiscoverAnonymousParticipant { name } } }
      }
      away {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant { profile { firstName lastName } }
            ... on DiscoverParticipantFillInPlayer { profile { firstName lastName } }
            ... on DiscoverGamePermitFillInPlayer { profile { firstName lastName } }
          }
          statistics { type { value } count }
        }
        periods { period { value } statistics { type { value } count } }
        bestPlayers { ranking participant { ... on DiscoverAnonymousParticipant { name } } }
      }
    }
  }
}
"""

def extract_period_scores(periods):
    quarter_map = {"FIRST_QTR": "Q1", "SECOND_QTR": "Q2", "THIRD_QTR": "Q3", "FOURTH_QTR": "Q4"}
    quarter_goals = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    quarter_behinds = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    quarter_scores = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for p in periods:
        q = quarter_map.get(p["period"]["value"])
        if not q:
            continue
        for s in p["statistics"]:
            if s["type"]["value"] == "TOTAL_SCORE":
                quarter_scores[q] = s["count"]
            elif s["type"]["value"] == "6_POINT_SCORE":
                quarter_goals[q] = s["count"]
            elif s["type"]["value"] == "1_POINT_SCORE":
                quarter_behinds[q] = s["count"]
    formatted = {}
    cg = cb = cs = 0
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        cg += quarter_goals[q]
        cb += quarter_behinds[q]
        cs += quarter_scores[q]
        formatted[q] = f"{cg}.{cb} ({cs})"
    return formatted

def extract_lineup(players):
    lineup = []
    for p in players:
        profile = p["player"].get("profile")
        if profile:
            lineup.append(f"#{p['playerNumber']} {profile['firstName']} {profile['lastName']}")
    return lineup

def extract_goal_scorers(players):
    scorers = []
    for p in players:
        profile = p["player"].get("profile")
        if not profile:
            continue
        goals = 0
        for s in p.get("statistics", []):
            if s["type"]["value"] == "6_POINT_SCORE":
                goals = s["count"]
                break
        if goals > 0:
            scorers.append({"name": f"{profile['firstName']} {profile['lastName']}", "goals": goals})
    scorers.sort(key=lambda x: x["goals"], reverse=True)
    return [f"{s['name']} ({s['goals']})" for s in scorers]

def extract_best_players(best_players_data):
    if not best_players_data:
        return []
    best = []
    try:
        for bp in sorted(best_players_data, key=lambda x: x.get("ranking", 999)):
            participant = bp.get("participant")
            if not participant or (isinstance(participant, dict) and not participant):
                continue
            name = participant.get("name") if isinstance(participant, dict) else participant
            if name:
                best.append(name)
    except:
        return []
    return best

def save_match_to_db(match):
    db = get_db()
    try:
        existing_match = db.query(Match).filter_by(match_id=match["match_id"]).first()
        if not existing_match:
            new_match = Match(
                match_id=match["match_id"],
                extracted_at=datetime.utcnow().isoformat(),
                date=match["date"],
                competition=match["competition"],
                venue=match["venue"],
                home_team=match["home_team"],
                away_team=match["away_team"],
                home_final_score=match["final_score"]["home"],
                away_final_score=match["final_score"]["away"],
                margin=abs(match["final_score"]["home"] - match["final_score"]["away"]),
                quarter_scores=json.dumps(match["period_scores"]),
                lineups=json.dumps(match["lineups"]),
                goal_scorers=json.dumps(match["goal_scorers"]),
                best_players=json.dumps(match["best_players"])
            )
            db.add(new_match)
            db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

def fetch_match_from_playhq(match_id: str) -> dict:
    payload = {"operationName": "gameView", "variables": {"gameId": match_id}, "query": GRAPHQL_QUERY}
    r = requests.post(PLAYHQ_GRAPHQL_URL, headers=HEADERS, json=payload, timeout=30)
    response = r.json()
    if "errors" in response:
        raise RuntimeError(response["errors"])
    game = response["data"]["discoverGame"]
    home_best = extract_best_players(game["statistics"]["home"].get("bestPlayers", [])) or None
    away_best = extract_best_players(game["statistics"]["away"].get("bestPlayers", [])) or None
    match = {
        "match_id": game["id"],
        "date": game["date"],
        "home_team": game["home"]["name"],
        "away_team": game["away"]["name"],
        "venue": game["allocation"]["court"]["venue"]["name"],
        "competition": game["round"]["grade"]["season"]["competition"]["name"],
        "final_score": {"home": game["result"]["home"]["score"], "away": game["result"]["away"]["score"]},
        "period_scores": {
            "home": extract_period_scores(game["statistics"]["home"]["periods"]),
            "away": extract_period_scores(game["statistics"]["away"]["periods"]),
        },
        "lineups": {
            "home": extract_lineup(game["statistics"]["home"]["players"]),
            "away": extract_lineup(game["statistics"]["away"]["players"]),
        },
        "goal_scorers": {
            "home": extract_goal_scorers(game["statistics"]["home"]["players"]),
            "away": extract_goal_scorers(game["statistics"]["away"]["players"]),
        },
        "best_players": {"home": home_best, "away": away_best}
    }
    save_match_to_db(match)
    return match

def build_match_knowledge(match: dict) -> str:
    home = match["home_team"]
    away = match["away_team"]
    fs_home = match["final_score"]["home"]
    fs_away = match["final_score"]["away"]
    margin = abs(fs_home - fs_away)
    hq = match["period_scores"]["home"]
    aq = match["period_scores"]["away"]
    home_scorers_text = ", ".join(match["goal_scorers"]["home"]) if match["goal_scorers"]["home"] else "None"
    away_scorers_text = ", ".join(match["goal_scorers"]["away"]) if match["goal_scorers"]["away"] else "None"
    home_best_text = "Not available" if match["best_players"]["home"] is None else (", ".join(match["best_players"]["home"]) or "None")
    away_best_text = "Not available" if match["best_players"]["away"] is None else (", ".join(match["best_players"]["away"]) or "None")
    return f"""
{home} played {away} in an Adelaide Footy League match.
The match took place on {match['date']} at {match['venue']} as part of the {match['competition']} season.

PERIOD SCORES (Cumulative Goals.Behinds):
End of Period | Q1        | Q2        | Q3        | Q4
{home}        | {hq['Q1']} | {hq['Q2']} | {hq['Q3']} | {hq['Q4']}
{away}        | {aq['Q1']} | {aq['Q2']} | {aq['Q3']} | {aq['Q4']}

Final score: {home} {fs_home} defeated {away} {fs_away}.
Margin: {margin} points.

Match Competitiveness Analysis:
- Final margin: {margin} points
- {"This was a close contest" if margin <= 20 else "This was a comfortable victory" if margin <= 40 else "This was a dominant performance"}

Team lineups:
{home}:
- {chr(10).join(match["lineups"]["home"])}

{away}:
- {chr(10).join(match["lineups"]["away"])}

GOAL SCORERS (OFFICIAL):
{home}: {home_scorers_text}
{away}: {away_scorers_text}

BEST PLAYERS (OFFICIAL):
{home}: {home_best_text}
{away}: {away_best_text}
""".strip()

def calculate_openai_cost(prompt_tokens, completion_tokens, model="gpt-4o-mini"):
    pricing = {
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "gpt-4o":      {"input": 2.50 / 1_000_000,  "output": 10.00 / 1_000_000},
    }
    p = pricing.get(model, pricing["gpt-4o-mini"])
    return prompt_tokens * p["input"] + completion_tokens * p["output"]

def save_generation_cost(user_id, match_id, content_type, prompt_tokens, completion_tokens, total_tokens, cost_usd, model):
    db = get_db()
    try:
        new_cost = GenerationCost(
            user_id=user_id, match_id=match_id, content_type=content_type,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total_tokens, cost_usd=cost_usd, model=model,
            generated_at=datetime.utcnow().isoformat()
        )
        db.add(new_cost)
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main_app():
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {st.session_state.user['username']}")
        st.markdown(f"**Role:** {st.session_state.user['role'].upper()}")
        st.divider()
        db = get_db()
        try:
            total_links = db.query(MatchLink).filter_by(is_active=1).count()
        finally:
            db.close()
        st.metric("📊 Available Matches", total_links)
        if "vectordb" in st.session_state:
            st.success("✅ Knowledge Base Active")
        else:
            st.info("ℹ️ Select a match to begin")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        st.divider()
        st.markdown("### 📚 Quick Guide")
        st.markdown("""
1. **Select** matches from dropdown
2. **Build** knowledge base
3. **Select** content type
4. **Generate** content
5. **Copy** and use!
        """)

    st.markdown("# 📰 Magazine Automation")
    st.markdown(f"### Welcome back, **{st.session_state.user['username']}**! 👋")
    st.divider()

    # --------------------------------------------------
    # Step 1: Select Matches
    # --------------------------------------------------
    st.markdown("## 🏈 Step 1: Select Matches")
    st.markdown("<br>", unsafe_allow_html=True)

    db = get_db()
    try:
        match_links = db.query(MatchLink).filter_by(is_active=1).order_by(MatchLink.date.desc()).all()
    finally:
        db.close()

    if not match_links:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 2rem; border-radius: 16px;
                    text-align: center; backdrop-filter: blur(10px);'>
            <h3 style='color: white;'>📭 No Matches Available Yet</h3>
            <p style='color: rgba(255,255,255,0.8);'>
                Your admin team hasn't added this week's matches yet. Please check back soon!
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build dropdown options grouped by competition
    competitions = {}
    for link in match_links:
        comp = link.competition or "Other"
        if comp not in competitions:
            competitions[comp] = []
        competitions[comp].append(link)

    match_map = {}
    dropdown_options = []
    for comp, comp_matches in competitions.items():
        for m in comp_matches:
            label = f"🏈 {m.home_team} vs {m.away_team}  ·  {m.date[:10] if m.date else 'TBD'}  ·  {comp}"
            dropdown_options.append(label)
            match_map[label] = m

    st.markdown("**Select one or more matches to generate content for:**")
    selected_labels = st.multiselect(
        "Available Matches",
        options=dropdown_options,
        placeholder="🔍 Search or select matches...",
        label_visibility="collapsed"
    )

    if not selected_labels:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📅 Available Matches")
        st.markdown("<br>", unsafe_allow_html=True)
        for comp, comp_matches in competitions.items():
            st.markdown(f"**🏆 {comp}**")
            cols = st.columns(min(len(comp_matches), 3))
            for i, m in enumerate(comp_matches):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 12px;
                                border: 1px solid rgba(255,255,255,0.2); margin-bottom: 1rem;'>
                        <p style='color: white; font-weight: 700; margin: 0; font-size: 1rem;'>{m.home_team}</p>
                        <p style='color: rgba(255,255,255,0.6); margin: 0.2rem 0; font-size: 0.85rem;'>vs</p>
                        <p style='color: white; font-weight: 700; margin: 0; font-size: 1rem;'>{m.away_team}</p>
                        <hr style='border-color: rgba(255,255,255,0.2); margin: 0.75rem 0;'>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;'>📅 {m.date[:10] if m.date else 'TBD'}</p>
                        <p style='color: rgba(255,255,255,0.8); margin: 0.2rem 0; font-size: 0.8rem;'>📍 {m.venue or 'TBD'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        st.info("👆 Use the dropdown above to select one or more matches, then click Build Knowledge Base.")
        return

    # --------------------------------------------------
    # Step 2: Build Knowledge Base
    # --------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🧠 Step 2: Build Knowledge Base")
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(min(len(selected_labels), 3))
    for i, label in enumerate(selected_labels):
        m = match_map[label]
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 12px;
                        border: 2px solid rgba(255,255,255,0.4); margin-bottom: 1rem;'>
                <p style='color: white; font-weight: 700; margin: 0; font-size: 1.05rem;'>
                    ✅ {m.home_team} vs {m.away_team}
                </p>
                <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
                    📅 {m.date[:10] if m.date else 'TBD'} · 📍 {m.venue or 'TBD'}
                </p>
                <p style='color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.8rem;'>
                    🏆 {m.competition}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        build_button = st.button("📥 Build Knowledge Base", use_container_width=True, type="primary")

    if build_button:
        with st.spinner("🔄 Fetching match data from PlayHQ..."):
            docs = []
            progress_bar = st.progress(0)
            for idx, label in enumerate(selected_labels):
                m = match_map[label]
                try:
                    match = fetch_match_from_playhq(m.match_id)
                    docs.append(Document(
                        page_content=build_match_knowledge(match),
                        metadata={"match_id": match["match_id"]}
                    ))
                    with st.expander(f"✅ {match['home_team']} vs {match['away_team']}", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Date:** {match['date']}")
                            st.markdown(f"**Venue:** {match['venue']}")
                        with c2:
                            st.markdown(f"**Score:** {match['final_score']['home']} – {match['final_score']['away']}")
                            st.markdown(f"**Margin:** {abs(match['final_score']['home'] - match['final_score']['away'])} pts")
                except Exception as e:
                    st.error(f"❌ Error fetching {m.home_team} vs {m.away_team}: {str(e)}")
                progress_bar.progress((idx + 1) / len(selected_labels))

            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = [
                    Document(page_content=c, metadata=d.metadata)
                    for d in docs
                    for c in splitter.split_text(d.page_content)
                ]
                st.session_state.vectordb = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings())
                st.session_state.selected_match_labels = selected_labels
                st.success(f"🎉 Knowledge base ready! {len(docs)} match(es), {len(chunks)} chunks indexed.")
                st.balloons()

    # --------------------------------------------------
    # Step 3: Generate Content
    # --------------------------------------------------
    if "vectordb" in st.session_state:
        st.divider()
        st.markdown("## ✍️ Step 3: Generate Content")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            content_type = st.selectbox(
                "📝 Content Type",
                ["Magazine match report", "Web article", "Social media long-form post"],
                index=0
            )
        with col2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            generate_button = st.button("🧠 Generate Content", use_container_width=True, type="primary")

        if generate_button:
            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name, temperature=0.25, max_tokens=1200)
            retriever = st.session_state.vectordb.as_retriever(k=6)

            magazine_prompt = """
You are a professional Australian football journalist writing for a print magazine.

CRITICAL RULES - READ CAREFULLY:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
3. Do NOT invent or guess any player names
4. Use the Period Scores table format exactly as shown in the context

LADDER POSITION RULES (STRICT):
- Ladder positions may be used ONLY if ladder data exists in database AND its date EXACTLY matches the match date being reported.
- If ladder data is from any earlier date, previous round, or is not explicitly tied to the same match date, DO NOT mention the ladder at all.
- When used, ladder positions must be stated factually and numerically only (e.g. "sitting second on the ladder").
- Do NOT describe form, momentum, struggles, dominance, or season trajectory unless explicitly stated in the provided data.
- Do NOT use subjective ladder language (e.g. "upper echelon", "struggling", "in form", "charging").
- If both teams have valid ladder positions for the same date, mention both.
- If only one team has valid ladder data for that date, mention ONLY that team.
- If ladder data is missing, outdated, or unclear, omit ladder references entirely.


OPENING PARAGRAPH - MUST BE CONTEXTUAL:
Look at the "Match Competitiveness Analysis" in the context to determine the tone:
- If margin ≤ 20 points: Use phrases like "In a closely fought contest", "In a tight encounter", or "In a thrilling clash"
- If margin 21-40 points: Use phrases like "In a solid performance", "In a commanding display", "In a professional showing", or "In a one sided match"
- If margin > 40 points: Use phrases like "In a dominant display", "In an emphatic victory", "In a comprehensive performance", or "In a one-sided affair"
- If margin > 90 points: Use phrases like "In an absolute mauling", "In a complete thrashing"

IMPORTANT: The opening MUST reflect the actual competitiveness of the match. Do NOT say "close encounter" if the margin was 50+ points!

STRUCTURE (USE EXACT HEADINGS):

1. Opening Paragraph (NO HEADING)
   - Start with contextually appropriate language based on margin
   - Mention venue and match context
   - Do NOT invent ladder positions, if ladder position exists in database of same data in which match data fetched, then use context to mention it like in first 
   paragraph mention team A is at at X position on ladder takes on team B  sitting at Y position.

2. Final Scores (EXACT HEADING)
   Use this exact table format:
   ```
   [Home Team]   | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
   [Away Team]   | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
   ```
   Use the exact cumulative scores from context (e.g., "5.3 (33)")

3. MATCH SUMMARY (EXACT HEADING)
   Write 4 paragraphs, one for each quarter:
   - Q1: Describe the first quarter action and end with the Q1 scoreline and margin
   - Q2: Describe the second quarter action and end with the Q2 scoreline and margin
   - Q3: Describe the third quarter action and end with the Q3 scoreline and margin
   - Q4: Describe the fourth quarter action and end with the Q4 scoreline and margin

4. FINAL WRAP-UP (EXACT HEADING)
   - Summarize how the winning team controlled the match
   - Mention key factors in the victory
   - State the final margin

5. BEST PLAYERS (EXACT HEADING)
   Use ONLY names from "BEST PLAYERS (OFFICIAL)":
   [Home Team]: [List official best players separated by commas]
   [Away Team]: [List official best players separated by commas]

6. GOAL SCORERS (EXACT HEADING)
   Use ONLY names from "GOAL SCORERS (OFFICIAL)":
   [Home Team]: [List official goal scorers with counts]
   [Away Team]: [List official goal scorers with counts]

7. PLAYED AT (EXACT HEADING)
   [Venue name]

STRICT RULES:
- Opening paragraph MUST match the actual margin (close/comfortable/dominant)
- Use ONLY official player names - NO inventions
- Use exact headings: "PERIOD SCORES", "MATCH SUMMARY", "FINAL WRAP-UP", "BEST PLAYERS", "GOAL SCORERS", "PLAYED AT"
- Editorial language is allowed, but all facts must come from context

LENGTH REQUIREMENT:
- Total: 750–900 words
- Do not exceed 950 words
- Do not go below 700 words

Context:
{context}

Write the magazine match report now.
"""

            web_article_prompt = """
You are a digital sports journalist writing an engaging web article for an online audience.

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
3. Do NOT invent or guess any player names

WEB ARTICLE STRUCTURE:

1. HEADLINE (Create a catchy, SEO-friendly headline)
   - Use action words and the winning team's name
   - Example: "[Team] Storm Home in Thrilling [Margin]-Point Victory"

2. LEAD PARAGRAPH (1-2 sentences)
   - Summarize the match result immediately
   - Include: Winner, loser, final score, venue, and what it means

3. KEY MOMENTS (Section heading)
   - Write 3-4 short paragraphs about crucial moments
   - Use subheadings for each moment (e.g., "First Quarter Blitz", "Second Half Comeback")
   - Make it scannable with bold text for key facts
   - Include specific scores and turning points

4. PLAYER PERFORMANCES (Section heading)
   - Highlight 2-3 standout players from OFFICIAL best players list
   - Write 1-2 sentences about each player's impact
   - Use quotes-style language: "X was instrumental" or "Y dominated"

5. THE STATS (Section heading)
   Present key statistics in bullet points:
   - Final Score: [Home] X defeated [Away] Y
   - Margin: Z points
   - Top Goal Scorers: [Use OFFICIAL list]
   - Best Players: [Use OFFICIAL list]
   - Venue: [Venue name]

6. WHAT IT MEANS (Section heading)
   - 1-2 paragraphs on implications
   - Keep it general - don't invent ladder positions
   - Focus on momentum, form, team confidence

WEB STYLE REQUIREMENTS:
- Short paragraphs (2-3 sentences max)
- Use subheadings frequently
- Bold important facts
- Active voice and present tense for immediacy
- Engaging, conversational tone
- SEO keywords: team names, venue, competition

LENGTH: 500-650 words

Context:
{context}

Write the web article now.
"""

            social_media_prompt = """
You are a social media content creator writing an engaging long-form post about an AFL match.

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
3. Do NOT invent or guess any player names

SOCIAL MEDIA LONG-FORM POST STRUCTURE:

1. ATTENTION-GRABBING OPENING (2-3 sentences)
   - Start with emoji and excitement
   - Announce the result with energy
   - Use conversational, enthusiastic tone
   Example: "🔥 WHAT. A. GAME! [Team] have done it again, defeating [Team] by [margin] points in an absolute thriller at [venue]!"

2. THE STORY (3-4 short paragraphs)
   - Tell the match narrative quarter by quarter
   - Use emojis strategically (⚡ 🎯 💪 🏆)
   - Keep sentences short and punchy
   - Build excitement and drama
   - Mention key moments and momentum shifts

3. THE HEROES (1-2 paragraphs)
   - Highlight 2-3 best players from OFFICIAL list
   - Use celebratory language
   - Give each player a 1-sentence shoutout
   Example: "Standing ovation for [Player] 👏 who was absolutely unstoppable with [X] goals!"

4. BY THE NUMBERS (Formatted list)
   Present as easy-to-read stats:
   
   📊 THE NUMBERS:
   ⚽ Final Score: [Home] X-Y [Away]
   📍 Venue: [Venue]
   ⭐ Margin: Z points
   🎯 Top Goal Kickers:
      • [Home team goals - use OFFICIAL list]
      • [Away team goals - use OFFICIAL list]
   💎 Best Players:
      • [Home team - use OFFICIAL list]
      • [Away team - use OFFICIAL list]

5. CLOSING HOOK (1-2 sentences)
   - End with forward-looking statement
   - Engage audience
   - Use relevant hashtags
   Example: "This team is on fire! 🔥 Can they keep this momentum going? Drop your thoughts below! 👇"

SOCIAL MEDIA STYLE REQUIREMENTS:
- Conversational, enthusiastic tone
- Use emojis (but don't overdo it - max 8-10 for entire post)
- Short paragraphs (1-3 sentences)
- Active voice, present tense
- Create FOMO and excitement
- Easy to read on mobile
- Include relevant hashtags at the end (3-5 hashtags)

EMOJI USAGE GUIDE:
- 🔥 (excitement, dominance)
- ⚡ (speed, momentum)
- 💪 (strength, performance)
- 🎯 (accuracy, goals)
- 🏆 (victory, excellence)
- 👏 (applause, recognition)
- 📊 (statistics)
- ⚽ (goals)
- 💎 (star players)

LENGTH: 350-500 words

Context:
{context}

Write the social media long-form post now. Remember to include hashtags at the end!
"""

            if content_type == "Magazine match report":
                prompt_text = magazine_prompt
            elif content_type == "Web article":
                prompt_text = web_article_prompt
            else:
                prompt_text = social_media_prompt

            prompt_template = ChatPromptTemplate.from_template(prompt_text)

            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            chain = (
                {"context": retriever | format_docs}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            with st.spinner("✨ Generating professional content..."):
                with get_openai_callback() as cb:
                    result = chain.invoke("Generate a match report")
                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens
                    total_tokens = cb.total_tokens
                    cost_usd = calculate_openai_cost(prompt_tokens, completion_tokens, model_name)

                    match_id = "unknown"
                    try:
                        docs_ret = retriever.get_relevant_documents("match")
                        if docs_ret:
                            match_id = docs_ret[0].metadata.get("match_id", "unknown")
                    except:
                        pass

                    save_generation_cost(
                        user_id=st.session_state.user['id'],
                        match_id=match_id,
                        content_type=content_type,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost_usd,
                        model=model_name
                    )

            st.markdown("## 📄 Generated Content")
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Word Count", len(result.split()))
            with col2:
                st.metric("📝 Characters", len(result))
            with col3:
                st.metric("📄 Type", content_type.split()[0])

            st.text_area("📋 Copy Text", result, height=300)
            st.success("✅ Content generated successfully!")

# --------------------------------------------------
# Run
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
