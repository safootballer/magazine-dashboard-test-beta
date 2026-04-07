import streamlit as st
import requests
import json
import hashlib
import os
import uuid
import re
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
# AUTO MIGRATION
# --------------------------------------------------
def run_migrations():
    try:
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        with engine.connect() as conn:
            if 'matches' in existing_tables:
                existing_cols = [c['name'] for c in inspector.get_columns('matches')]
                patches = {
                    'best_players':   'ALTER TABLE matches ADD COLUMN best_players TEXT',
                    'margin':         'ALTER TABLE matches ADD COLUMN margin INTEGER',
                    'lineups':        'ALTER TABLE matches ADD COLUMN lineups TEXT',
                    'goal_scorers':   'ALTER TABLE matches ADD COLUMN goal_scorers TEXT',
                    'quarter_scores': 'ALTER TABLE matches ADD COLUMN quarter_scores TEXT',
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
# Page Config  ← UPDATED: SAFie branding
# --------------------------------------------------
st.set_page_config(
    page_title="SAFie | AI by SA Footballer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
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
    .publish-box {
        background: rgba(255,255,255,0.15); padding: 2rem; border-radius: 16px;
        border: 2px solid rgba(255,255,255,0.3); backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    /* ── SAFie branding badge ── */
    .safie-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.92);
        text-transform: uppercase;
        margin-top: 6px;
    }
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
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span { color: #ffffff !important; }
    .stSuccess { color: #065f46 !important; }
    .stInfo    { color: #1e3a8a !important; }
    .stWarning { color: #78350f !important; }
    .stError   { color: #7f1d1d !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Logo — dual logo header  ← UPDATED
# Displays:  [SA Footballer logo]  |  SAFie wordmark  |  [SAFie logo]
# Falls back gracefully if either image file is missing.
# --------------------------------------------------
def render_logo_center():
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Three columns: left logo | centre text | right logo
    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_left:
        # SAFie logo — new logo on the LEFT (assets/logo2.png)
        try:
            st.image("assets/logo2.png", width=150)
        except Exception:
            st.markdown(
                "<p style='color:white; font-size:0.8rem; text-align:center;'>SAFie</p>",
                unsafe_allow_html=True
            )

    with col_mid:
        # App name + tagline centred between the two logos
        st.markdown("""
        <div style="text-align:center; padding-top: 12px;">
            <h1 style="margin:0; font-size:2.6rem; font-weight:800;
                       color:white; letter-spacing:-0.02em; line-height:1.1;">
                SAFie
            </h1>
            <span class="safie-badge">AI by SA Footballer</span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # SA Footballer logo — existing logo on the RIGHT (assets/logo.png)
        try:
            st.image("assets/logo.png", width=150)
        except Exception:
            st.markdown(
                "<p style='color:white; font-size:0.8rem; text-align:center;'>SA Footballer</p>",
                unsafe_allow_html=True
            )

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
# Login Page  ← UPDATED: SAFie branding
# --------------------------------------------------
def login_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1 style="margin: 0; font-size: 3rem; color: white; font-weight: 800;">SAFie</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; margin-top: 0.4rem; color: white; opacity: 0.9;">AI by SA Footballer</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.95rem; margin-top: 0.2rem; color: rgba(255,255,255,0.7);">AI-Powered Match Report Generation</p>', unsafe_allow_html=True)
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
    for key in ["vectordb", "generated_content", "generated_content_type",
                "publish_success", "selected_match_labels"]:
        if key in st.session_state:
            del st.session_state[key]
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
    competition = match["competition"]
    home_scorers_text = ", ".join(match["goal_scorers"]["home"]) if match["goal_scorers"]["home"] else "None"
    away_scorers_text = ", ".join(match["goal_scorers"]["away"]) if match["goal_scorers"]["away"] else "None"
    home_best_text = "Not available" if match["best_players"]["home"] is None else (", ".join(match["best_players"]["home"]) or "None")
    away_best_text = "Not available" if match["best_players"]["away"] is None else (", ".join(match["best_players"]["away"]) or "None")
    return f"""
{home} played {away} in a {competition} match.
The match took place on {match['date']} at {match['venue']} as part of the {competition} season.

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
# Sanity Publisher
# --------------------------------------------------
COMPETITION_MAP = {
    "AFL": "AFL",
    "AFLW": "AFLW",
    "SANFL": "SANFL",
    "SANFLW": "SANFLW",
    "Amateur": "Amateur",
    "Amateurs": "Amateur",
    "SAWFL Women's": "SAWFL Women's",
    "Country Football": "Country Football",
}

COUNTRY_LEAGUES = {
    "Adelaide Plains": "adelaide-plains",
    "Barossa Light & Gawler": "barossa",
    "Broken Hill": "broken-hill",
    "Eastern Eyre": "eastern-eyre",
    "Far North": "far-north",
    "Great Flinders": "great-flinders",
    "Great Southern": "great-southern",
    "Hills Division 1": "hills-div1",
    "Hills Country Division": "hills-country",
    "Kangaroo Island": "kangaroo-island",
    "Kowree Naracoorte Tatiara": "knt",
    "Limestone Coast": "limestone-coast",
    "Murray Valley": "murray-valley",
    "Mid South Eastern": "mid-south-eastern",
    "North Eastern": "north-eastern",
    "Northern Areas": "northern-areas",
    "Port Lincoln": "port-lincoln",
    "River Murray": "river-murray",
    "Riverland": "riverland",
    "Southern": "southern",
    "Spencer Gulf": "spencer-gulf",
    "Western Eyre": "western-eyre",
    "Whyalla": "whyalla",
    "Yorke Peninsula": "yorke-peninsula",
}

PLAYHQ_TO_COUNTRY_LEAGUE = {
    "Adelaide Plains Football League": "adelaide-plains",
    "Barossa Light & Gawler Football League": "barossa",
    "Broken Hill Football League": "broken-hill",
    "Eastern Eyre Football League": "eastern-eyre",
    "Far North Football League": "far-north",
    "Great Flinders Football League": "great-flinders",
    "Great Southern Football League": "great-southern",
    "Hills Division 1 Football League": "hills-div1",
    "Hills Country Division Football League": "hills-country",
    "Kangaroo Island Football League": "kangaroo-island",
    "Kowree Naracoorte Tatiara Football League": "knt",
    "Limestone Coast Football League": "limestone-coast",
    "Murray Valley Football League": "murray-valley",
    "Mid South Eastern Football League": "mid-south-eastern",
    "North Eastern Football League": "north-eastern",
    "Northern Areas Football League": "northern-areas",
    "Port Lincoln Football League": "port-lincoln",
    "River Murray Football League": "river-murray",
    "Riverland Football League": "riverland",
    "Southern Football League": "southern",
    "Spencer Gulf Football League": "spencer-gulf",
    "Western Eyre Football League": "western-eyre",
    "Whyalla Football League": "whyalla",
    "Yorke Peninsula Football League": "yorke-peninsula",
}


AUTHORS = [
    "Ethan Parker", "Caleb Murphy", "Dylan Fraser", "Blake Henderson",
    "Nathan Collins", "Connor Walsh", "Jordan Hughes", "Ryan McCarthy",
    "Mitchell Dawson", "Jake Sullivan", "Tyler Bennett", "Corey Richards",
    "Ben Lawson", "Josh McLean", "Kyle Donovan", "Aaron Griffiths",
    "Sam Peterson", "Luke Davidson", "Bailey Thornton", "Trent Gallagher",
    "Liam O'Connor", "Noah Williams", "Oliver Smith", "William Brown",
    "James Taylor", "Lucas Wilson", "Henry Anderson", "Alexander Clark",
    "Charlie Walker", "Mason Hall", "Cooper Allen", "Hudson Young",
    "Hunter King", "Riley Scott", "Zachary Green", "Isaac Adams",
    "Max Baker", "Harry Mitchell", "Archie Carter", "Charlotte Johnson",
    "Olivia White", "Amelia Harris", "Isla Martin", "Mia Thompson",
    "Ava Robinson", "Grace Lee", "Chloe Walker", "Ella Wright",
    "Emily Scott", "Harper King", "Sophie Turner", "Evie Collins",
    "Ruby Stewart", "Willow Morris", "Zoe Bell", "Matilda Cooper",
    "Lily Ward", "Hannah Brooks", "Lucy Bennett", "Poppy Sanders",
    "Aria Jenkins", "Layla Price", "Scarlett Murphy", "Ellie Kelly",
    "Jacklyn Davies", "Brooke Sullivan", "Tahlia McKenzie", "Paige Donnelly",
    "Tayla Fitzgerald",
]


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text[:96]


def text_to_portable_text(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    blocks = []
    for para in paragraphs:
        blocks.append({
            "_type": "block",
            "_key": uuid.uuid4().hex[:12],
            "style": "normal",
            "markDefs": [],
            "children": [{
                "_type": "span",
                "_key": uuid.uuid4().hex[:12],
                "text": para,
                "marks": []
            }]
        })
    return blocks


def extract_excerpt_from_content(content_text, content_type):
    lines = [l.strip() for l in content_text.split('\n') if l.strip()]
    for line in lines:
        if len(line) > 80 and not line.startswith('#') and not line.startswith('**') and not line.startswith('|'):
            return line[:300]
    return lines[0][:300] if lines else ""


def publish_to_sanity(title, slug, competition, excerpt, content_text, author, country_league=None, as_draft=False):
    project_id = os.getenv("SANITY_PROJECT_ID") or st.secrets.get("SANITY_PROJECT_ID", "")
    dataset    = os.getenv("SANITY_DATASET", "production") or st.secrets.get("SANITY_DATASET", "production")
    token      = os.getenv("SANITY_TOKEN") or st.secrets.get("SANITY_TOKEN", "")

    if not project_id or not token:
        return False, "❌ Sanity credentials not found. Add SANITY_PROJECT_ID and SANITY_TOKEN to your .env or secrets."

    doc_id = f"{'drafts.' if as_draft else ''}editorial-{uuid.uuid4().hex[:16]}"
    url = f"https://{project_id}.api.sanity.io/v2024-01-01/data/mutate/{dataset}"

    doc = {
        "_id": doc_id,
        "_type": "editorial",
        "title": title,
        "slug": {"_type": "slug", "current": slug},
        "competition": competition,
        "excerpt": excerpt,
        "content": text_to_portable_text(content_text),
        "author": author,
        "publishedAt": datetime.utcnow().isoformat() + "Z",
    }

    if competition == "Country Football" and country_league:
        doc["countryLeague"] = country_league

    payload = {"mutations": [{"createOrReplace": doc}]}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            return True, slug
        else:
            return False, f"Sanity returned status {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


# --------------------------------------------------
# Facebook Publisher
# --------------------------------------------------
def clean_for_facebook(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'__(.*?)__',     r'\1', text)
    text = re.sub(r'_(.*?)_',       r'\1', text)

    replacements = [
        (r'(?im)^#+\s*ATTENTION[^\n]*',        '🔥 '),
        (r'(?im)^#+\s*THE STORY[^\n]*',        '📖 THE STORY'),
        (r'(?im)^#+\s*THE HEROES[^\n]*',       '⭐ THE HEROES'),
        (r'(?im)^#+\s*BY THE NUMBERS[^\n]*',   '📊 BY THE NUMBERS'),
        (r'(?im)^#+\s*CLOSING[^\n]*',          '👇 '),
        (r'(?im)^#+\s*KEY MOMENTS[^\n]*',      '⚡ KEY MOMENTS'),
        (r'(?im)^#+\s*PLAYER PERFORMANCES[^\n]*', '💪 PLAYER PERFORMANCES'),
        (r'(?im)^#+\s*THE STATS[^\n]*',        '📊 THE STATS'),
        (r'(?im)^#+\s*WHAT IT MEANS[^\n]*',    '🏆 WHAT IT MEANS'),
        (r'(?im)^#+\s*HEADLINE[^\n]*',         ''),
        (r'(?im)^#+\s*',                        ''),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def post_to_facebook(message, image_bytes=None, image_name=None):
    page_id    = os.getenv("FACEBOOK_PAGE_ID") or st.secrets.get("FACEBOOK_PAGE_ID", "")
    page_token = os.getenv("FACEBOOK_PAGE_TOKEN") or st.secrets.get("FACEBOOK_PAGE_TOKEN", "")

    if not page_id or not page_token:
        return False, "❌ Facebook credentials not found. Add FACEBOOK_PAGE_ID and FACEBOOK_PAGE_TOKEN to your .env"

    try:
        if image_bytes:
            upload_url = f"https://graph.facebook.com/v19.0/{page_id}/photos"
            files      = {"source": (image_name or "photo.jpg", image_bytes, "image/jpeg")}
            upload_data = {
                "published":  "false",
                "no_story":   "true",
                "access_token": page_token,
            }
            upload_resp = requests.post(upload_url, data=upload_data, files=files, timeout=30)
            upload_result = upload_resp.json()

            if "id" not in upload_result:
                error_msg = upload_result.get("error", {}).get("message", str(upload_result))
                return False, f"Photo upload failed: {error_msg}"

            photo_id = upload_result["id"]

            feed_url  = f"https://graph.facebook.com/v19.0/{page_id}/feed"
            feed_data = {
                "message":           message,
                "attached_media[0]": json.dumps({"media_fbid": photo_id}),
                "access_token":      page_token,
            }
            resp = requests.post(feed_url, data=feed_data, timeout=15)

        else:
            url  = f"https://graph.facebook.com/v19.0/{page_id}/feed"
            resp = requests.post(url, data={"message": message, "access_token": page_token}, timeout=15)

        result = resp.json()
        if "id" in result:
            post_id  = result["id"]
            post_url = f"https://www.facebook.com/{page_id}/posts/{post_id.split('_')[-1]}"
            return True, post_url
        else:
            error_msg = result.get("error", {}).get("message", str(result))
            return False, f"Facebook API error: {error_msg}"

    except Exception as e:
        return False, str(e)


# --------------------------------------------------
# Main App  ← UPDATED: SAFie branding in sidebar & header
# --------------------------------------------------
def main_app():
    with st.sidebar:
        # Sidebar dual-logo (smaller)
        try:
            sb_col1, sb_col2 = st.columns(2)
            with sb_col1:
                st.image("assets/logo2.png", width=80)
            with sb_col2:
                st.image("assets/logo.png", width=80)
        except Exception:
            st.markdown("### 🏈 SAFie")

        st.markdown("""
        <div style="text-align:center; margin: 4px 0 12px 0;">
            <span style="color:white; font-size:1.1rem; font-weight:700;">SAFie</span><br>
            <span style="color:rgba(255,255,255,0.65); font-size:0.72rem; letter-spacing:0.06em;">
                AI BY SA FOOTBALLER
            </span>
        </div>
        """, unsafe_allow_html=True)

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
        if "generated_content" in st.session_state:
            st.success("✅ Content Ready to Publish")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        st.divider()
        st.markdown("### 📚 Quick Guide")
        st.markdown("""
1. **Select** matches from dropdown
2. **Build** knowledge base
3. **Generate** content
4. **Publish** live to website
        """)

    # ── Main header ──────────────────────────────────────────────
    st.markdown("# 🏈 SAFie — AI by SA Footballer")
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

                first_match = match_map[selected_labels[0]]
                raw_comp = first_match.competition or "AFL"
                st.session_state.current_competition = raw_comp

                detected_league = PLAYHQ_TO_COUNTRY_LEAGUE.get(raw_comp)
                if detected_league:
                    st.session_state.current_competition = "Country Football"
                    st.session_state.detected_country_league = detected_league
                else:
                    st.session_state.detected_country_league = None

                st.success(f"🎉 Knowledge base ready! {len(docs)} match(es), {len(chunks)} chunks indexed.")
                st.balloons()

    # --------------------------------------------------
    # Step 3: Generate Content
    # --------------------------------------------------
    if "vectordb" in st.session_state:
        st.divider()
        st.markdown("## ✍️ Step 3: Generate Content")
        st.markdown("<br>", unsafe_allow_html=True)

        if "content_type_selection" not in st.session_state:
            st.session_state.content_type_selection = "Magazine match report"

        col1, col2 = st.columns([2, 1])
        with col1:
            content_type = st.selectbox(
                "📝 Content Type",
                ["Magazine match report", "Web article", "Social media long-form post"],
                index=["Magazine match report", "Web article", "Social media long-form post"].index(
                    st.session_state.content_type_selection
                ),
                key="content_type_select"
            )
            st.session_state.content_type_selection = content_type

        with col2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            generate_button = st.button("🧠 Generate Content", use_container_width=True, type="primary")

        if generate_button:
            for k in ["publish_success", "published_slug", "generated_content", "generated_content_type"]:
                if k in st.session_state:
                    del st.session_state[k]

            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name, temperature=0.25, max_tokens=1200)
            retriever = st.session_state.vectordb.as_retriever(k=6)

            magazine_prompt = """
You are a professional Australian football journalist writing for a print magazine.
Write in Australian English throughout — use Australian spelling and expressions (e.g. "colour" not "color", "organisation" not "organization", "centre" not "center", "defence" not "defense", "practise" not "practice", "travelled" not "traveled").

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
- If margin <= 20 points: Use phrases like "In a closely fought contest", "In a tight encounter", or "In a thrilling clash"
- If margin 21-40 points: Use phrases like "In a solid performance", "In a commanding display", "In a professional showing", or "In a one sided match"
- If margin > 40 points: Use phrases like "In a dominant display", "In an emphatic victory", "In a comprehensive performance", or "In a one-sided affair"
- If margin > 90 points: Use phrases like "In an absolute mauling", "In a complete thrashing"

IMPORTANT: The opening MUST reflect the actual competitiveness of the match.

STRUCTURE (USE EXACT HEADINGS):
1. Opening Paragraph (NO HEADING)
2. Final Scores (EXACT HEADING)
   [Home Team]   | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
   [Away Team]   | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
3. MATCH SUMMARY (EXACT HEADING) - 4 paragraphs, one per quarter
4. FINAL WRAP-UP (EXACT HEADING)
5. BEST PLAYERS (EXACT HEADING)
6. GOAL SCORERS (EXACT HEADING)
7. PLAYED AT (EXACT HEADING)

LENGTH REQUIREMENT: 750-900 words

Context:
{context}

Write the magazine match report now.
"""

            web_article_prompt = """
You are a digital sports journalist writing an engaging web article for an online audience.
Write in Australian English throughout — use Australian spelling and expressions (e.g. "colour" not "color", "organisation" not "organization", "centre" not "center", "defence" not "defense").

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
3. Do NOT invent or guess any player names
4. Always refer to the competition by its actual name from the context — never call it "Adelaide Footy League" unless that is literally the competition name in the context.

WEB ARTICLE STRUCTURE:
1. HEADLINE
2. LEAD PARAGRAPH
3. KEY MOMENTS (Section heading)
4. PLAYER PERFORMANCES (Section heading)
5. THE STATS (Section heading)
6. WHAT IT MEANS (Section heading)

LENGTH: 500-650 words

Context:
{context}

Write the web article now.
"""

            social_media_prompt = """
You are a social media content creator writing an engaging long-form post about an Australian football match.
Write in Australian English throughout — use Australian spelling and expressions (e.g. "colour" not "color", "organisation" not "organization", "centre" not "center", "defence" not "defense").

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
3. Do NOT invent or guess any player names
4. Always refer to the competition by its actual name from the context.

SOCIAL MEDIA POST STRUCTURE:
1. ATTENTION-GRABBING OPENING — start with a strong emoji and punchy sentence
2. THE STORY (quarter by quarter) — use ⚡ Q1, ⚡ Q2 etc to label each quarter
3. THE HEROES — use ⭐ to highlight each player
4. BY THE NUMBERS — use 📊 and bullet points with emojis (e.g. 🏆 Final Score, 🎯 Top Scorers)
5. CLOSING HOOK + hashtags

IMPORTANT FORMATTING RULES:
- Do NOT use **asterisks** for bold — Facebook renders them as literal characters
- Do NOT use any markdown formatting at all
- Use emojis to make headings and key facts stand out instead
- Keep paragraphs short — 2-3 sentences max

LENGTH: 350-500 words

Context:
{context}

Write the social media long-form post now.
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

            st.session_state.generated_content = result
            st.session_state.generated_content_type = content_type
            st.rerun()

        # Show generated content
        if "generated_content" in st.session_state:
            result = st.session_state.generated_content
            saved_content_type = st.session_state.get("generated_content_type", "")

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
                st.metric("📄 Type", saved_content_type.split()[0] if saved_content_type else "")

            st.text_area("📋 Copy Text", result, height=300)
            st.success("✅ Content generated successfully!")

            # Facebook posting — only shown for social media posts
            if saved_content_type == "Social media long-form post":
                st.divider()
                st.markdown("### 📘 Post to Facebook Page")
                st.markdown("""
                <div style='background: rgba(255,255,255,0.12); padding: 1rem; border-radius: 10px;
                            border: 1px solid rgba(255,255,255,0.2); margin-bottom: 1rem;'>
                    <p style='color: white; margin: 0; font-size: 0.9rem;'>
                        Review and edit the post below, optionally attach a photo, then click Post.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                fb_text = st.text_area(
                    "✏️ Edit post before sending (optional)",
                    value=clean_for_facebook(result),
                    height=220,
                    key="fb_post_text"
                )

                fb_char_count = len(fb_text)
                if fb_char_count > 63206:
                    st.warning(f"⚠️ Post is {fb_char_count} characters — Facebook limit is 63,206.")

                st.markdown("**📷 Attach a photo (optional)**")
                include_photo = st.radio(
                    "Include a photo?",
                    ["No photo — text only", "Yes — upload a photo"],
                    index=0,
                    horizontal=True,
                    key="fb_include_photo"
                )

                fb_photo = None
                if include_photo == "Yes — upload a photo":
                    fb_photo = st.file_uploader(
                        "Choose an image",
                        type=["jpg", "jpeg", "png"],
                        key="fb_photo_upload",
                        help="JPG or PNG. Will be posted alongside your text."
                    )
                    if fb_photo:
                        st.image(fb_photo, caption="Preview", width=300)

                col_fb1, col_fb2, col_fb3 = st.columns([1, 2, 1])
                with col_fb2:
                    fb_button = st.button(
                        "📘 Post to Facebook Now",
                        use_container_width=True,
                        key="fb_post_btn",
                        type="primary",
                        disabled=(fb_char_count > 63206)
                    )

                if fb_button:
                    with st.spinner("📡 Posting to Facebook..."):
                        fb_success, fb_result = post_to_facebook(
                            fb_text,
                            image_bytes=fb_photo.read() if fb_photo else None,
                            image_name=fb_photo.name if fb_photo else None
                        )
                    if fb_success:
                        st.success("🎉 **Posted to Facebook Page!**")
                        st.info(f"🔗 View post: {fb_result}")
                    else:
                        st.error(fb_result)

    # --------------------------------------------------
    # Step 4: Publish to Website
    # --------------------------------------------------
    if "generated_content" in st.session_state:
        st.divider()
        st.markdown("## 🚀 Step 4: Publish to SA Footballer Website")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.get("publish_success"):
            published_slug = st.session_state.get("published_slug", "")
            st.success("🎉 **Article is LIVE on your website!**")
            st.info(f"🔗 View it at: `https://sa-footballer-website.vercel.app/editorials/{published_slug}`")
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: rgba(255,255,255,0.12); padding: 1.5rem; border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.25); margin-bottom: 1rem;'>
            <p style='color: white; margin: 0; font-size: 0.95rem;'>
                Fill in the details below and hit <strong>Publish Live</strong> —
                your article will appear on the Editorials page immediately. No Sanity Studio needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            first_line = st.session_state.generated_content.split('\n')[0].strip()
            first_line = re.sub(r'^#+\s*', '', first_line)
            first_line = re.sub(r'\*+', '', first_line)

            pub_title = st.text_input(
                "Article Title *",
                value=first_line[:120] if len(first_line) > 10 else "",
                placeholder="e.g. Glenelg Dominate in 45-Point Victory Over Sturt",
                key="pub_title"
            )

            auto_slug = slugify(pub_title) if pub_title else ""
            pub_slug = st.text_input(
                "Slug (URL path) *",
                value=auto_slug,
                help="Auto-generated from title. This becomes the article URL.",
                key="pub_slug"
            )

            auto_excerpt = extract_excerpt_from_content(
                st.session_state.generated_content,
                st.session_state.get("generated_content_type", "")
            )
            pub_excerpt = st.text_area(
                "Excerpt * (shown on editorial cards)",
                value=auto_excerpt[:300],
                height=100,
                key="pub_excerpt"
            )

            pub_author = st.selectbox(
                "Author",
                AUTHORS,
                index=0,
                key="pub_author"
            )

        with col2:
            competition_options = ["AFL", "AFLW", "SANFL", "SANFLW", "Amateur", "SAWFL Women's", "Country Football"]
            raw_comp = st.session_state.get("current_competition", "AFL")
            mapped_comp = COMPETITION_MAP.get(raw_comp, "AFL")
            default_idx = competition_options.index(mapped_comp) if mapped_comp in competition_options else 0

            pub_competition = st.selectbox(
                "Competition *",
                competition_options,
                index=default_idx,
                key="pub_competition"
            )

            pub_country_league = None
            if pub_competition == "Country Football":
                detected = st.session_state.get("detected_country_league")
                league_keys = list(COUNTRY_LEAGUES.keys())
                league_values = list(COUNTRY_LEAGUES.values())
                default_league_idx = league_values.index(detected) if detected and detected in league_values else 0

                league_name = st.selectbox(
                    "Country League *",
                    league_keys,
                    index=default_league_idx,
                    key="pub_league"
                )
                pub_country_league = COUNTRY_LEAGUES[league_name]

            st.markdown("<br>", unsafe_allow_html=True)

            ready = bool(pub_title and pub_slug and pub_excerpt)
            if not ready:
                st.warning("⚠️ Fill in Title, Slug and Excerpt to enable publishing.")

            st.markdown("<br>", unsafe_allow_html=True)

            publish_live = st.button(
                "🚀 Publish Live Now",
                type="primary",
                use_container_width=True,
                disabled=not ready,
                key="publish_live_btn"
            )

            save_draft = st.button(
                "💾 Save as Draft",
                use_container_width=True,
                disabled=not bool(pub_title),
                key="save_draft_btn",
                help="Saves to Sanity Studio only — won't appear on website until published from Studio."
            )

        if publish_live:
            with st.spinner("📡 Publishing to SA Footballer website..."):
                success, result_msg = publish_to_sanity(
                    title=pub_title,
                    slug=pub_slug,
                    competition=pub_competition,
                    excerpt=pub_excerpt,
                    content_text=st.session_state.generated_content,
                    author=pub_author,
                    country_league=pub_country_league,
                    as_draft=False
                )
            if success:
                st.session_state.publish_success = True
                st.session_state.published_slug = result_msg
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ Publish failed: {result_msg}")

        if save_draft:
            with st.spinner("💾 Saving draft to Sanity..."):
                success, result_msg = publish_to_sanity(
                    title=pub_title,
                    slug=pub_slug,
                    competition=pub_competition,
                    excerpt=pub_excerpt,
                    content_text=st.session_state.generated_content,
                    author=pub_author,
                    country_league=pub_country_league,
                    as_draft=True
                )
            if success:
                st.success("💾 Draft saved in Sanity Studio. Go to Studio to review and publish.")
            else:
                st.error(f"❌ Draft save failed: {result_msg}")


# --------------------------------------------------
# Run
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
