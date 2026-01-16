import streamlit as st
import requests
import sqlite3
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# Init
# --------------------------------------------------
load_dotenv()

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

/* REMOVE DEFAULT STREAMLIT TOP GAP */
section[data-testid="stAppViewContainer"] {
    padding-top: 0.5rem !important;
}

/* REMOVE EMPTY WHITE BLOCKS (IMPORTANT FIX) */
div[data-testid="stVerticalBlock"]:empty {
    display: none !important;
}

div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #059669 0%, #a3e635 100%);
}

/* LOGO WRAPPER */
.logo-wrapper {
    margin-top: 0.5rem;
    margin-bottom: 1.2rem;
}

/* LOGIN CARD */
.login-card {
    background: rgba(255, 255, 255, 0.96);
    padding: 3rem;
    border-radius: 24px;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.25);
    max-width: 460px;
    margin: 1.5rem auto 3rem auto;
}

/* REMOVE ANY CONTAINER BACKGROUND */
.stContainer {
    background: transparent !important;
}

/* TITLES */
.big-title {
    font-size: 3.1rem;
    font-weight: 800;
    text-align: center;
    color: #064e3b;
    margin-bottom: 0.25rem;
}

.subtitle {
    text-align: center;
    color: #065f46;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #065f46 0%, #16a34a 100%);
    color: white;
    border: none;
    padding: 0.85rem;
    border-radius: 14px;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 25px rgba(6, 95, 70, 0.4);
}

/* INPUTS */
input, textarea {
    border-radius: 12px !important;
}

/* CONTENT CARD */
.content-card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 18px;
    border: none;
    box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    font-family: 'Georgia', serif;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CENTER LOGO (RENDER AFTER CSS)
# --------------------------------------------------
def render_logo_center():
    st.markdown('<div class="logo-wrapper">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo.png", width=170)
    st.markdown('</div>', unsafe_allow_html=True)

# 🔥 Render logo AFTER CSS
render_logo_center()


# --------------------------------------------------
# Database setup
# --------------------------------------------------
conn = sqlite3.connect("magazine.db", check_same_thread=False)
cur = conn.cursor()

# Matches table
cur.execute("""
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE,
    extracted_at TEXT,
    date TEXT,
    competition TEXT,
    venue TEXT,
    home_team TEXT,
    away_team TEXT,
    home_final_score INTEGER,
    away_final_score INTEGER,
    margin INTEGER,
    quarter_scores TEXT,
    lineups TEXT,
    goal_scorers TEXT
)
""")

# Users table
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT,
    last_login TEXT
)
""")

# Create default admin user if not exists
def create_default_admin():
    admin_password = hashlib.sha256("admin123".encode()).hexdigest()
    try:
        cur.execute("""
            INSERT INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?)
        """, ("admin", admin_password, "admin", datetime.utcnow().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass

create_default_admin()
conn.commit()

# --------------------------------------------------
# Authentication Functions
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    password_hash = hash_password(password)
    result = cur.execute("""
        SELECT id, username, role FROM users 
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash)).fetchone()
    
    if result:
        cur.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.utcnow().isoformat(), result[0]))
        conn.commit()
        return {"id": result[0], "username": result[1], "role": result[2]}
    return None

def login_page():
    # Hero section
    st.markdown('<div style="text-align: center; padding: 2rem 0;">', unsafe_allow_html=True)
    st.markdown("# 📰")
    st.markdown('<h1 class="big-title">Sports Magazine Automation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Match Report Generation</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Login card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        st.markdown("### 🔐 Welcome Back")
        st.markdown("Sign in to generate professional match reports")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("ℹ️ Default Credentials"):
            st.code("Username: admin\nPassword: admin123")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 3rem;">⚡</div>
            <div class="stat-label">Fast</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">Generate reports in seconds</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 3rem;">🎯</div>
            <div class="stat-label">Accurate</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">Powered by AI technology</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 3rem;">✍️</div>
            <div class="stat-label">Professional</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">Magazine-quality content</div>
        </div>
        """, unsafe_allow_html=True)

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
    home { ... on DiscoverTeam { name } }
    away { ... on DiscoverTeam { name } }

    allocation { court { venue { name } } }

    round {
      grade {
        season {
          competition { name }
        }
      }
    }

    result {
      home { score }
      away { score }
    }

    statistics {
      home {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant { profile { firstName lastName } }
            ... on DiscoverParticipantFillInPlayer { profile { firstName lastName } }
            ... on DiscoverGamePermitFillInPlayer { profile { firstName lastName } }
          }
          statistics {
            type { value }
            count
          }
        }
        periods {
          period { value }
          statistics { type { value } count }
        }
      }

      away {
        players {
          playerNumber
          player {
            ... on DiscoverParticipant { profile { firstName lastName } }
            ... on DiscoverParticipantFillInPlayer { profile { firstName lastName } }
            ... on DiscoverGamePermitFillInPlayer { profile { firstName lastName } }
          }
          statistics {
            type { value }
            count
          }
        }
        periods {
          period { value }
          statistics { type { value } count }
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
    quarter_map = {
        "FIRST_QTR": "Q1",
        "SECOND_QTR": "Q2",
        "THIRD_QTR": "Q3",
        "FOURTH_QTR": "Q4",
    }

    scores = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}

    for p in periods:
        q = quarter_map.get(p["period"]["value"])
        if not q:
            continue
        for s in p["statistics"]:
            if s["type"]["value"] == "TOTAL_SCORE":
                scores[q] = s["count"]

    return scores

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
            scorers.append(f"{name} ({goals})")

    return scorers

def save_match_to_db(match):
    cur.execute("""
        INSERT OR IGNORE INTO matches (
            match_id, extracted_at, date, competition, venue,
            home_team, away_team, home_final_score, away_final_score,
            margin, quarter_scores, lineups, goal_scorers
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match["match_id"],
        datetime.utcnow().isoformat(),
        match["date"],
        match["competition"],
        match["venue"],
        match["home_team"],
        match["away_team"],
        match["final_score"]["home"],
        match["final_score"]["away"],
        abs(match["final_score"]["home"] - match["final_score"]["away"]),
        json.dumps(match["period_scores"]),
        json.dumps(match["lineups"]),
        json.dumps(match["goal_scorers"]),
    ))
    conn.commit()

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

    return f"""
{home} played {away} in an Adelaide Footy League match.

The match took place on {match['date']} at {match['venue']} as part of the {match['competition']} season.

Quarter-by-quarter scores:
Q1: {home} {hq['Q1']} – {away} {aq['Q1']}
Q2: {home} {hq['Q2']} – {away} {aq['Q2']}
Q3: {home} {hq['Q3']} – {away} {aq['Q3']}
Q4: {home} {hq['Q4']} – {away} {aq['Q4']}

Final score:
{home} {fs_home} defeated {away} {fs_away}.

Margin: {margin} points.

Team lineups:

{home}:
- {"\n- ".join(match["lineups"]["home"])}

{away}:
- {"\n- ".join(match["lineups"]["away"])}

GOAL SCORERS:

{home}:
{home_scorers_text}

{away}:
{away_scorers_text}
""".strip()

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
        total_matches = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
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
                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
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
            st.markdown("<br>", unsafe_allow_html=True)
            generate_button = st.button("🧠 Generate Content", use_container_width=True, type="primary")

        if generate_button:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.25,
                max_tokens=1200
            )

            retriever = st.session_state.vectordb.as_retriever(k=6)
            
            prompt_text = """
You are a professional Australian football journalist writing for a print magazine.

You MUST follow this structure exactly.

STRUCTURE:

1. Opening Paragraph
- Mention venue and match context
- Describe relative team strength using neutral editorial tone
- Do NOT invent ladder positions unless provided

2. QUARTER BY QUARTER SCORES
- List Q1, Q2, Q3, Q4
- Use exact scorelines from context

3. Quarter-by-Quarter Match Narrative
- One paragraph per quarter (Q1, Q2, Q3, Q4)
- Mention which team controlled the quarter
- Quote the exact scoreline after each quarter
- Mention the margin after each quarter

4. Final Summary Paragraph
- Summarise how the winning team controlled the match
- Mention the final margin

5. BEST PLAYERS
- List best players for each team separately
- Use player names from context only

6. GOAL SCORERS SECTION (MANDATORY):
- List goal scorers for each team separately
- Use names exactly as provided in context
- Include goal counts as shown in context
- Do NOT invent scorers

7. PLAYED AT
- Write the venue name

STRICT RULES:
- Use ONLY the provided context
- Do NOT invent players, scores, or facts
- Editorial language is allowed, facts are not
- Formatting must match magazine style

LENGTH REQUIREMENT:
- Total article length: 750–900 words
- Do not exceed 950 words
- Do not go below 700 words

Context: {context}

Write the magazine match report now.
"""

            prompt_template = ChatPromptTemplate.from_template(prompt_text)
            
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])
            
            # Modern chain using LCEL
            chain = (
                {"context": retriever | format_docs}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            with st.spinner("✨ Generating professional content..."):
                result = chain.invoke("Generate a match report")
                
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
