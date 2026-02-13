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

def create_default_admin():
    db = get_db()
    try:
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
    finally:
        db.close()

create_default_admin()

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
        return None
    finally:
        db.close()

def login_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📰")
    st.markdown('<h1 style="margin: 0; font-size: 3rem;">Sports Magazine Automation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; margin-top: 0.5rem;">AI-Powered Match Report Generation</p>', unsafe_allow_html=True)
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

def extract_match_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]

def extract_period_scores(periods):
    quarter_map = {
        "FIRST_QTR": "Q1",
        "SECOND_QTR": "Q2",
        "THIRD_QTR": "Q3",
        "FOURTH_QTR": "Q4",
    }
    
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
            
            name = None
            if isinstance(participant, dict):
                name = participant.get("name")
            elif isinstance(participant, str):
                name = participant
            
            if name:
                best.append(name)
    except (KeyError, TypeError, AttributeError):
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
    payload = {
        "operationName": "gameView",
        "variables": {"gameId": match_id},
        "query": GRAPHQL_QUERY,
    }
    r = requests.post(PLAYHQ_GRAPHQL_URL, headers=HEADERS, json=payload, timeout=30)
    response = r.json()
    if "errors" in response:
        raise RuntimeError(response["errors"])

    game = response["data"]["discoverGame"]

    home_best = extract_best_players(game["statistics"]["home"].get("bestPlayers", []))
    away_best = extract_best_players(game["statistics"]["away"].get("bestPlayers", []))

    if not home_best:
        home_best = None
    
    if not away_best:
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
    pricing = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,
            "output": 0.600 / 1_000_000,
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,
            "output": 10.00 / 1_000_000,
        },
        "gpt-4-turbo": {
            "input": 10.00 / 1_000_000,
            "output": 30.00 / 1_000_000,
        }
    }
    
    if model not in pricing:
        model = "gpt-4o-mini"
    
    input_cost = prompt_tokens * pricing[model]["input"]
    output_cost = completion_tokens * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return total_cost

def save_generation_cost(user_id, match_id, content_type, prompt_tokens, completion_tokens, total_tokens, cost_usd, model):
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
    finally:
        db.close()

def main_app():
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {st.session_state.user['username']}")
        st.markdown(f"**Role:** {st.session_state.user['role'].upper()}")
        st.divider()
        
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

    st.markdown("# 📰 Magazine Automation")
    st.markdown(f"### Welcome back, **{st.session_state.user['username']}**! 👋")
    st.divider()

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
            llm = ChatOpenAI(model=model_name, temperature=0.25, max_tokens=1200)
            retriever = st.session_state.vectordb.as_retriever(k=6)

            magazine_prompt = """
You are a professional Australian football journalist writing for a print magazine.

CRITICAL RULES - READ CAREFULLY:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. If best players show "Not available", write exactly: "Best players not available"
3. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
4. Do NOT invent or guess any player names
5. Use the Period Scores table format exactly as shown in the context

OPENING PARAGRAPH - MUST BE CONTEXTUAL:
Look at the "Match Competitiveness Analysis" in the context to determine the tone:
- If margin ≤ 20 points: Use phrases like "In a closely fought contest", "In a tight encounter", or "In a thrilling clash"
- If margin 21-40 points: Use phrases like "In a solid performance", "In a commanding display", "In a professional showing", or "In a one sided match"
- If margin > 40 points: Use phrases like "In a dominant display", "In an emphatic victory", "In a comprehensive performance", or "In a one-sided affair"

STRUCTURE (USE EXACT HEADINGS):

1. Opening Paragraph (NO HEADING) - MUST BE 2-3 SENTENCES
   - First sentence: Start with contextually appropriate language based on margin, mention venue, state the result
   - Second sentence: Add context about the match (team performances, key factors, turning points)
   - Optional third sentence: Additional match context or significance
   - Mention ONLY the venue (e.g., "at [Venue Name]")
   - Do NOT mention the date
   - Do NOT mention the competition name

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

LENGTH REQUIREMENT: 750–900 words

Context:
{context}

Write the magazine match report now.
"""

            web_article_prompt = """
You are a digital sports journalist writing an engaging web article for an online audience.

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. If best players show "Not available", write exactly: "Best players not available"
3. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
4. Do NOT invent or guess any player names

WEB ARTICLE STRUCTURE:

1. HEADLINE (Create a catchy, SEO-friendly headline)

2. LEAD PARAGRAPH (1-2 sentences)

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
You are a social media content creator writing an engaging long-form post about an AFL match.

CRITICAL RULES:
1. Use ONLY the exact best players listed in the "BEST PLAYERS (OFFICIAL)" section
2. If best players show "Not available", write exactly: "Best players not available"
3. Use ONLY the exact goal scorers listed in the "GOAL SCORERS (OFFICIAL)" section
4. Do NOT invent or guess any player names

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
                        docs = retriever.get_relevant_documents("match")
                        if docs:
                            match_id = docs[0].metadata.get("match_id", "unknown")
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

            word_count = len(result.split())
            char_count = len(result)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Word Count", word_count)
            with col2:
                st.metric("📝 Characters", char_count)
            with col3:
                st.metric("📄 Type", content_type.split()[0])

            st.text_area("📋 Copy Text", result, height=300)
            st.success("✅ Content generated successfully!")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
