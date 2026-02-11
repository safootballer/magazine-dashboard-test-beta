import streamlit as st
import requests
import json
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
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
# Get database URL from environment variable (Render will provide this)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')

# Fix for Render PostgreSQL URL (they use postgres:// but SQLAlchemy needs postgresql://)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()

# Define models
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

# Create tables
Base.metadata.create_all(engine)

# Create session factory
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

st.set_page_config(
    page_title="Sports Magazine Automation",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS (INJECT FIRST)
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem 0;
    }
    .login-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CENTER LOGO (RENDER AFTER CSS)
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

# 🔥 Render logo AFTER CSS
render_logo_center()

# --------------------------------------------------
# Create default admin user if not exists
# --------------------------------------------------
def create_default_admin():
    db = get_db()
    try:
        # Check if admin exists
        existing_admin = db.query(User).filter_by(username="admin").first()
        if not existing_admin:
            admin_password = hashlib.sha256("admin123".encode()).hexdigest()
            admin_user = User(
                username="admin",
                password_hash=admin_password,
                role="admin",
                created_at=datetime.utcnow().isoformat()
            )
            db.add(admin_user)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error creating admin: {e}")
    finally:
        db.close()

create_default_admin()

# --------------------------------------------------
# Authentication Functions
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
    except Exception as e:
        print(f"Login error: {e}")
        return None
    finally:
        db.close()

def login_page():
    # Hero section
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📰")
    st.markdown('<h1 style="margin: 0; font-size: 3rem;">Sports Magazine Automation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; margin-top: 0.5rem;">AI-Powered Match Report Generation</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Login card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome Back")
        st.markdown("Sign in to generate professional match reports")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        st.markdown("&nbsp;", unsafe_allow_html=True)
        
        if st.button("🚀 Sign In"):
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
        
        st.markdown("&nbsp;", unsafe_allow_html=True)
        
        with st.expander("ℹ️ Default Credentials"):
            st.code("Username: admin\nPassword: admin123")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("&nbsp;", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""<div class="feature-card">
            <h2>⚡</h2>
            <h4>Fast</h4>
            <p>Generate reports in seconds</p>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="feature-card">
            <h2>🎯</h2>
            <h4>Accurate</h4>
            <p>Powered by AI technology</p>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""<div class="feature-card">
            <h2>✍️</h2>
            <h4>Professional</h4>
            <p>Magazine-quality content</p>
        </div>""", unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    if "vectordb" in st.session_state:
        del st.session_state.vectordb
    st.rerun()

# --------------------------------------------------
# PlayHQ GraphQL config
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
    home {
      ... on DiscoverTeam {
        name
      }
    }
    away {
      ... on DiscoverTeam {
        name
      }
    }
    allocation {
      court {
        venue {
          name
        }
      }
    }
    round {
      grade {
        season {
          competition {
            name
          }
        }
      }
    }
    result {
      home {
        score
      }
      away {
        score
      }
    }
    statistics {
      home {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant {
              profile {
                firstName
                lastName
              }
            }
            ... on DiscoverParticipantFillInPlayer {
              profile {
                firstName
                lastName
              }
            }
            ... on DiscoverGamePermitFillInPlayer {
              profile {
                firstName
                lastName
              }
            }
          }
          statistics {
            type {
              value
            }
            count
          }
        }
        periods {
          period {
            value
          }
          statistics {
            type {
              value
            }
            count
          }
        }
        bestPlayers {
          ranking
          participant {
            ... on DiscoverAnonymousParticipant {
              name
            }
          }
        }
      }
      away {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant {
              profile {
                firstName
                lastName
              }
            }
            ... on DiscoverParticipantFillInPlayer {
              profile {
                firstName
                lastName
              }
            }
            ... on DiscoverGamePermitFillInPlayer {
              profile {
                firstName
                lastName
              }
            }
          }
          statistics {
            type {
              value
            }
            count
          }
        }
        periods {
          period {
            value
          }
          statistics {
            type {
              value
            }
            count
          }
        }
        bestPlayers {
          ranking
          participant {
            ... on DiscoverAnonymousParticipant {
              name
            }
          }
        }
      }
    }
  }
}
"""

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def extract_match_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]

def extract_period_scores(periods):
    """Extract cumulative scores at end of each quarter in Goals.Behinds (Total) format"""
    quarter_map = {
        "FIRST_QTR": "Q1",
        "SECOND_QTR": "Q2",
        "THIRD_QTR": "Q3",
        "FOURTH_QTR": "Q4",
    }
    
    # Per-quarter values from API
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
    
    # Calculate cumulative values (PlayHQ style)
    formatted = {}
    cumulative_goals = 0
    cumulative_behinds = 0
    cumulative_score = 0
    
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        cumulative_goals += quarter_goals[q]
        cumulative_behinds += quarter_behinds[q]
        cumulative_score += quarter_scores[q]
        formatted[q] = f"{cumulative_goals}.{cumulative_behinds} ({cumulative_score})"
    
    return formatted

def extract_lineup(players):
    lineup = []
    for p in players:
        profile = p["player"].get("profile")
        if profile:
            lineup.append(
                f"#{p['playerNumber']} {profile['firstName']} {profile['lastName']}"
            )
    return lineup

def extract_goal_scorers(players):
    """Extract goal scorers and sort by goals in descending order"""
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
            name = f"{profile['firstName']} {profile['lastName']}"
            scorers.append({"name": name, "goals": goals})
    
    # Sort by goals in descending order (highest first)
    scorers.sort(key=lambda x: x["goals"], reverse=True)
    
    # Format as "Name (goals)"
    return [f"{s['name']} ({s['goals']})" for s in scorers]

def extract_best_players(best_players_data):
    """Extract best players from the bestPlayers field - OFFICIAL DATA with debug logging"""
    print(f"🔍 DEBUG - Best players raw data: {best_players_data}")
    
    if not best_players_data:
        print("⚠️ No best players data provided")
        return []
    
    best = []
    try:
        for idx, bp in enumerate(sorted(best_players_data, key=lambda x: x.get("ranking", 999))):
            print(f"🔍 DEBUG - Processing best player #{idx + 1}: {bp}")
            participant = bp.get("participant")
            
            if not participant or (isinstance(participant, dict) and not participant):
                print(f"⚠️ Empty participant data for entry #{idx + 1}")
                continue
            
            # Try to get name
            name = None
            if isinstance(participant, dict):
                name = participant.get("name")
                print(f"🔍 DEBUG - Found name in participant dict: {name}")
            elif isinstance(participant, str):
                name = participant
                print(f"🔍 DEBUG - Participant is string: {name}")
            
            if name:
                best.append(name)
    except (KeyError, TypeError, AttributeError) as e:
        print(f"❌ Error extracting best players: {e}")
        return []
    
    print(f"✅ Final best players list: {best}")
    return best

def save_match_to_db(match):
    db = get_db()
    try:
        # Check if match already exists
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
        print(f"Error saving match: {e}")
    finally:
        db.close()

def fetch_match_from_playhq(match_id: str) -> dict:
    payload = {
        "operationName": "gameView",
        "variables": {"gameId": match_id},
        "query": GRAPHQL_QUERY,
    }
    r = requests.post(
        PLAYHQ_GRAPHQL_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )
    response = r.json()
    if "errors" in response:
        raise RuntimeError(response["errors"])

    game = response["data"]["discoverGame"]

    # DEBUG: Print raw best players data
    print("🔍 DEBUG - Raw HOME bestPlayers from API:", game["statistics"]["home"].get("bestPlayers"))
    print("🔍 DEBUG - Raw AWAY bestPlayers from API:", game["statistics"]["away"].get("bestPlayers"))

    # Extract best players from API
    home_best = extract_best_players(game["statistics"]["home"].get("bestPlayers", []))
    away_best = extract_best_players(game["statistics"]["away"].get("bestPlayers", []))

    # If no best players data available, set as None (will be handled in display)
    if not home_best:
        print("⚠️ No HOME best players from API - marking as unavailable")
        home_best = None
    
    if not away_best:
        print("⚠️ No AWAY best players from API - marking as unavailable")
        away_best = None

    match = {
        "match_id": game["id"],
        "date": game["date"],
        "home_team": game["home"]["name"],
        "away_team": game["away"]["name"],
        "venue": game["allocation"]["court"]["venue"]["name"],
        "competition": game["round"]["grade"]["season"]["competition"]["name"],
        "final_score": {
            "home": game["result"]["home"]["score"],
            "away": game["result"]["away"]["score"],
        },
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
        "best_players": {
            "home": home_best,
            "away": away_best,
        }
    }

    print(f"🔍 DEBUG - Final match best_players: HOME={match['best_players']['home']}, AWAY={match['best_players']['away']}")
    save_match_to_db(match)
    return match

def build_match_knowledge(match: dict) -> str:
    """Build enhanced match knowledge with official best players and formatted scores"""
    home = match["home_team"]
    away = match["away_team"]
    fs_home = match["final_score"]["home"]
    fs_away = match["final_score"]["away"]
    margin = abs(fs_home - fs_away)
    hq = match["period_scores"]["home"]
    aq = match["period_scores"]["away"]

    home_scorers_text = ", ".join(match["goal_scorers"]["home"]) if match["goal_scorers"]["home"] else "None"
    away_scorers_text = ", ".join(match["goal_scorers"]["away"]) if match["goal_scorers"]["away"] else "None"
    
    # Handle None for best players
    if match["best_players"]["home"] is None:
        home_best_text = "Not available"
    else:
        home_best_text = ", ".join(match["best_players"]["home"]) if match["best_players"]["home"] else "None"
    
    if match["best_players"]["away"] is None:
        away_best_text = "Not available"
    else:
        away_best_text = ", ".join(match["best_players"]["away"]) if match["best_players"]["away"] else "None"

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
    """
    Calculate the cost of OpenAI API usage based on token counts.
    
    Pricing as of 2024 (update these if pricing changes):
    GPT-4o-mini:
    - Input: $0.150 per 1M tokens
    - Output: $0.600 per 1M tokens
    
    GPT-4o:
    - Input: $2.50 per 1M tokens  
    - Output: $10.00 per 1M tokens
    """
    
    pricing = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,   # $0.150 per 1M tokens
            "output": 0.600 / 1_000_000,  # $0.600 per 1M tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,    # $2.50 per 1M tokens
            "output": 10.00 / 1_000_000,  # $10.00 per 1M tokens
        },
        "gpt-4-turbo": {
            "input": 10.00 / 1_000_000,   # $10.00 per 1M tokens
            "output": 30.00 / 1_000_000,  # $30.00 per 1M tokens
        }
    }
    
    if model not in pricing:
        model = "gpt-4o-mini"  # Default fallback
    
    input_cost = prompt_tokens * pricing[model]["input"]
    output_cost = completion_tokens * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return total_cost

def save_generation_cost(user_id, match_id, content_type, prompt_tokens, completion_tokens, total_tokens, cost_usd, model):
    """Save generation cost to database"""
    db = get_db()
    try:
        new_cost = GenerationCost(
            user_id=user_id,
            match_id=match_id,
            content_type=content_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model=model,
            generated_at=datetime.utcnow().isoformat()
        )
        db.add(new_cost)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error saving cost: {e}")
    finally:
        db.close()

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {st.session_state.user['username']}")
        st.markdown(f"**Role:** {st.session_state.user['role'].upper()}")
        st.divider()
        
        # Quick stats
        db = get_db()
        try:
            total_matches = db.query(Match).count()
        finally:
            db.close()
        
        st.metric("📊 Total Matches", total_matches)
        
        if "vectordb" in st.session_state:
            st.success("✅ Knowledge Base Active")
        else:
            st.info("ℹ️ No Knowledge Base")
        
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.divider()
        st.markdown("### 📚 Quick Guide")
        st.markdown("""
1. **Paste** PlayHQ match URLs
2. **Build** knowledge base
3. **Generate** content
4. **Copy** and use!
        """)

    # Main header
    st.markdown("# 📰 Magazine Automation")
    st.markdown(f"### Welcome back, **{st.session_state.user['username']}**! 👋")
    st.divider()

    # Step 1: Input URLs
    st.markdown("## 🔗 Step 1: Add Match URLs")
    st.markdown("Paste PlayHQ match URLs below (one per line)")
    
    urls = st.text_area(
        "Match URLs",
        placeholder="https://www.playhq.com/afl/org/adelaide-footy-league/game/...\nhttps://www.playhq.com/afl/org/adelaide-footy-league/game/...",
        height=150,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        build_button = st.button("📥 Build Knowledge Base", use_container_width=True, type="primary")
    
    if build_button:
        if not urls.strip():
            st.error("⚠️ Please enter at least one URL")
        else:
            with st.spinner("🔄 Fetching match data from PlayHQ..."):
                docs = []
                progress_bar = st.progress(0)
                url_list = [u.strip() for u in urls.splitlines() if u.strip()]
                
                for idx, url in enumerate(url_list):
                    try:
                        match_id = extract_match_id(url)
                        match = fetch_match_from_playhq(match_id)
                        docs.append(
                            Document(
                                page_content=build_match_knowledge(match),
                                metadata={"match_id": match["match_id"]}
                            )
                        )
                        
                        # Show success in an expander
                        with st.expander(f"✅ {match['home_team']} vs {match['away_team']}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Date:** {match['date']}")
                                st.markdown(f"**Venue:** {match['venue']}")
                            with col2:
                                st.markdown(f"**Score:** {match['final_score']['home']} - {match['final_score']['away']}")
                                st.markdown(f"**Margin:** {abs(match['final_score']['home'] - match['final_score']['away'])} pts")
                    
                    except Exception as e:
                        st.error(f"❌ Error fetching {url}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(url_list))
                
                if docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = [
                        Document(page_content=c, metadata=d.metadata)
                        for d in docs
                        for c in splitter.split_text(d.page_content)
                    ]
                    st.session_state.vectordb = InMemoryVectorStore.from_documents(
                        chunks, OpenAIEmbeddings()
                    )
                    st.success(f"🎉 Knowledge base created! {len(docs)} matches, {len(chunks)} chunks")
                    st.balloons()

    # Step 2: Generate Content
    if "vectordb" in st.session_state:
        st.divider()
        st.markdown("## ✍️ Step 2: Generate Content")
        
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
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.25,
                max_tokens=1200
            )
            retriever = st.session_state.vectordb.as_retriever(k=6)

            # [The rest of the prompts remain the same - I'll include the magazine_prompt as example]
            magazine_prompt = """
You are a professional Australian football journalist writing for a print magazine.

CRITICAL RULES - READ CAREFULLY:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. If best players show "Not available", write exactly: "Best players not available"
3. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
4. Do NOT invent or guess any player names
5. Use the Period Scores table format exactly as shown in the context

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

1. Opening Paragraph (NO HEADING) - MUST BE 2-3 SENTENCES
   - First sentence: Start with contextually appropriate language based on margin, mention venue, state the result
   - Second sentence: Add context about the match (team performances, key factors, turning points)
   - Optional third sentence: Additional match context or significance
   - Mention ONLY the venue (e.g., "at [Venue Name]")
   - Do NOT mention the date
   - Do NOT mention the competition name (e.g., Adelaide Footy League)
   - If ladder positions exist in database for same date as match, mention them naturally in the opening sentences
   
   Examples:
   - "In a dominant display at Payneham Oval, Port District defeated Golden Grove by 51 points. The home side controlled the contest from start to finish, with their forwards proving too strong for the opposition defense."
   
   - "In a closely fought contest at West Lakes, Glenelg sitting third on the ladder edged out Sturt by 12 points. The match lived up to expectations with both teams trading blows throughout the four quarters."

2. Final Scores (EXACT HEADING)
   Use this exact table format:
```
   [Home Team] | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
   [Away Team] | [Q1 score] | [Q2 score] | [Q3 score] | [Q4 score]
```
   Use the exact cumulative scores from context (e.g., "5.3 (33)")

3. MATCH SUMMARY (EXACT HEADING)
   Write 4 paragraphs, one for each quarter:
   - 1: Describe the first quarter action and end with the Q1 scoreline and margin
   - 2: Describe the second quarter action and end with the Q2 scoreline and margin
   - 3: Describe the third quarter action and end with the Q3 scoreline and margin
   - 4: Describe the fourth quarter action and end with the Q4 scoreline and margin

4. FINAL WRAP-UP (EXACT HEADING)
   - Summarize how the winning team controlled the match
   - Mention key factors in the victory
   - State the final margin

5. BEST PLAYERS (EXACT HEADING)
   If best players are listed, use them:
   [Home Team]: [List official best players separated by commas]
   [Away Team]: [List official best players separated by commas]
   
   If best players show "Not available", write:
   [Home Team]: Best players not available
   [Away Team]: Best players not available

6. GOAL SCORERS (EXACT HEADING)
   Use ONLY names from "GOAL SCORERS (OFFICIAL)":
   [Home Team]: [List official goal scorers with counts]
   [Away Team]: [List official goal scorers with counts]

7. PLAYED AT (EXACT HEADING)
   [Venue name]

STRICT RULES:
- Opening paragraph MUST be 2-3 sentences (NOT just one sentence)
- Opening paragraph MUST match the actual margin (close/comfortable/dominant)
- Opening paragraph mentions ONLY venue - NO date, NO competition name
- Use ONLY official player names - NO inventions
- Use exact headings: "Final Scores", "MATCH SUMMARY", "FINAL WRAP-UP", "BEST PLAYERS", "GOAL SCORERS", "PLAYED AT"
- Editorial language is allowed, but all facts must come from context

LENGTH REQUIREMENT:
- Total: 750–900 words
- Do not exceed 950 words
- Do not go below 700 words

Context:
{context}

Write the magazine match report now.
"""

            # [Include web_article_prompt and social_media_prompt here - same as before]
            
            web_article_prompt = """[Same as in previous response]"""
            social_media_prompt = """[Same as in previous response]"""

            # Select the appropriate prompt based on content type
            if content_type == "Magazine match report":
                prompt_text = magazine_prompt
            elif content_type == "Web article":
                prompt_text = web_article_prompt
            else:  # Social media long-form post
                prompt_text = social_media_prompt

            prompt_template = ChatPromptTemplate.from_template(prompt_text)

            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            # Modern LCEL chain
            chain = (
                {"context": retriever | format_docs}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            with st.spinner("✨ Generating professional content..."):
                # Use the LLM with token tracking
                with get_openai_callback() as cb:
                    result = chain.invoke("Generate a match report")
                    
                    # Extract token usage
                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens
                    total_tokens = cb.total_tokens
                    
                    # Calculate cost
                    cost_usd = calculate_openai_cost(prompt_tokens, completion_tokens, model_name)
                    
                    # Get match_id from the first document in vectordb (if available)
                    match_id = "unknown"
                    try:
                        # Retrieve one document to get match_id from metadata
                        docs = retriever.get_relevant_documents("match")
                        if docs:
                            match_id = docs[0].metadata.get("match_id", "unknown")
                    except:
                        pass
                    
                    # Save cost to database
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
            
            # Display in a nice card
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)

            # Stats
            word_count = len(result.split())
            char_count = len(result)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Word Count", word_count)
            with col2:
                st.metric("📝 Characters", char_count)
            with col3:
                st.metric("📄 Type", content_type.split()[0])

            # Copy button
            st.text_area("📋 Copy Text", result, height=300)
            
            st.success("✅ Content generated successfully!")

# --------------------------------------------------
# Run App
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
