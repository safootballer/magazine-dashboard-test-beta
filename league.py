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
# DATABASE SETUP
# --------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()

# --------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------
class League(Base):
    """Saved leagues that get auto-synced daily."""
    __tablename__ = 'leagues'
    id = Column(Integer, primary_key=True, autoincrement=True)
    grade_id = Column(String(255), unique=True, nullable=False)
    grade_name = Column(String(255))
    season = Column(String(255))
    url = Column(Text)
    added_by = Column(String(255))
    added_at = Column(String(255))
    last_synced_at = Column(String(255))
    sync_enabled = Column(Integer, default=1)  # 1 = enabled, 0 = disabled

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
    points_for = Column(Integer, default=0)
    points_against = Column(Integer, default=0)
    forfeits = Column(Integer, default=0)
    synced_at = Column(String(255))

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(String(255))
    last_login = Column(String(255))

Base.metadata.create_all(engine)

# --------------------------------------------------
# MIGRATION
# --------------------------------------------------
def migrate_database():
    inspector = inspect(engine)

    if 'ladder' in inspector.get_table_names():
        existing = [c['name'] for c in inspector.get_columns('ladder')]
        new_cols = [
            ("byes",           "INTEGER DEFAULT 0"),
            ("points_for",     "INTEGER DEFAULT 0"),
            ("points_against", "INTEGER DEFAULT 0"),
            ("forfeits",       "INTEGER DEFAULT 0"),
        ]
        for col_name, col_def in new_cols:
            if col_name not in existing:
                try:
                    with engine.connect() as conn:
                        conn.execute(text(f"ALTER TABLE ladder ADD COLUMN {col_name} {col_def}"))
                        conn.commit()
                except Exception as e:
                    print(f"⚠️ Could not add {col_name}: {e}")

migrate_database()

SessionLocal = sessionmaker(bind=engine)

def get_db():
    return SessionLocal()

# --------------------------------------------------
# PLAYHQ
# --------------------------------------------------
PLAYHQ_URL = "https://api.playhq.com/graphql"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Origin": "https://www.playhq.com",
    "Referer": "https://www.playhq.com/",
    "tenant": "afl"
}

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
        played won lost drawn byes
        competitionPoints alternatePercentage
        pointsFor pointsAgainst forfeits
        team { id name }
      }
    }
  }
}
"""

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
        r = requests.post(PLAYHQ_URL, headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("data")
    except Exception:
        return None

def fetch_grade_meta(grade_id):
    data = safe_post({"query": GRADE_META_QUERY, "variables": {"gradeID": grade_id}})
    if not data:
        return None
    g = data["discoverGrade"]
    return {"id": g["id"], "name": g["name"], "season": g["season"]["name"]}

def sync_ladder(grade_id, grade_name, season, silent=False):
    """Sync ladder for a single grade. silent=True suppresses st.* calls (for cron use)."""
    synced_at = datetime.utcnow().isoformat()
    data = safe_post({"query": LADDER_QUERY, "variables": {"gradeID": grade_id}})
    if not data:
        if not silent:
            st.error("❌ Failed to fetch ladder from PlayHQ")
        return False

    grade = data.get("discoverGrade")
    if not grade or not grade.get("ladder"):
        if not silent:
            st.warning("No ladder data returned")
        return False

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

                vals = dict(
                    grade_id=grade_id,
                    grade_name=grade_name,
                    round_name=round_name,
                    team_name=row["team"]["name"],
                    rank=idx,
                    played=row["played"],
                    wins=row["won"],
                    losses=row["lost"],
                    draws=row["drawn"],
                    byes=row.get("byes", 0),
                    points=row["competitionPoints"],
                    percentage=row["alternatePercentage"],
                    points_for=row.get("pointsFor", 0),
                    points_against=row.get("pointsAgainst", 0),
                    forfeits=row.get("forfeits", 0),
                    synced_at=synced_at
                )

                if existing:
                    for k, v in vals.items():
                        setattr(existing, k, v)
                else:
                    db.add(Ladder(
                        team_id=row["team"]["id"],
                        season=season,
                        round_id=round_id,
                        **vals
                    ))

        # Update last_synced_at in leagues table
        league = db.query(League).filter_by(grade_id=grade_id).first()
        if league:
            league.last_synced_at = synced_at

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        if not silent:
            st.error(f"Database error: {e}")
        return False
    finally:
        db.close()

def sync_all_leagues(silent=False):
    """Sync all enabled leagues. Called by cron or admin manually."""
    db = get_db()
    try:
        leagues = db.query(League).filter_by(sync_enabled=1).all()
        results = []
        for league in leagues:
            ok = sync_ladder(league.grade_id, league.grade_name, league.season, silent=silent)
            results.append((league.grade_name, ok))
        return results
    finally:
        db.close()

# --------------------------------------------------
# AUTH
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_login(username, password):
    db = get_db()
    try:
        user = db.query(User).filter_by(
            username=username,
            password_hash=hash_password(password)
        ).first()
        if user:
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    finally:
        db.close()

def is_admin():
    return st.session_state.get("user", {}).get("role") == "admin"

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    h1, h2, h3 { color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #334155 100%); }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stDataFrame { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .stButton>button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 12px; font-weight: 600; }
    div[data-testid="metric-container"] { background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    .league-card { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align:center;'>🏈 League Info Login</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("### 🔐 Sign In")
        username = st.text_input("👤 Username")
        password = st.text_input("🔒 Password", type="password")
        if st.button("🚀 Login", use_container_width=True, type="primary"):
            if username and password:
                user = verify_login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            else:
                st.warning("⚠️ Enter both fields")

# --------------------------------------------------
# SIDEBAR — LEAGUE MANAGEMENT (admin only)
# --------------------------------------------------
def sidebar_league_manager():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['username']} ({st.session_state.user['role'].upper()})")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

        st.divider()

        if is_admin():
            st.markdown("### ➕ Add League")
            new_url = st.text_input("PlayHQ Grade URL", placeholder="https://www.playhq.com/.../grade/abc123")
            if st.button("Add & Sync", use_container_width=True, type="primary"):
                grade_id = extract_grade_id(new_url)
                if not grade_id:
                    st.error("Invalid URL")
                else:
                    meta = fetch_grade_meta(grade_id)
                    if not meta:
                        st.error("Could not fetch league info")
                    else:
                        db = get_db()
                        try:
                            existing = db.query(League).filter_by(grade_id=grade_id).first()
                            if existing:
                                st.warning("League already saved")
                            else:
                                db.add(League(
                                    grade_id=grade_id,
                                    grade_name=meta["name"],
                                    season=meta["season"],
                                    url=new_url,
                                    added_by=st.session_state.user["username"],
                                    added_at=datetime.utcnow().isoformat(),
                                    sync_enabled=1
                                ))
                                db.commit()
                                sync_ladder(grade_id, meta["name"], meta["season"])
                                st.success(f"✅ Added: {meta['name']}")
                                st.rerun()
                        finally:
                            db.close()

            st.divider()
            st.markdown("### 🔄 Sync All Now")
            if st.button("Sync All Leagues", use_container_width=True):
                with st.spinner("Syncing all leagues..."):
                    results = sync_all_leagues()
                for name, ok in results:
                    if ok:
                        st.success(f"✅ {name}")
                    else:
                        st.error(f"❌ {name}")
                st.rerun()

        st.divider()
        st.markdown("### 🏆 Leagues")
        db = get_db()
        try:
            leagues = db.query(League).order_by(League.grade_name).all()
            if not leagues:
                st.info("No leagues added yet")
            for lg in leagues:
                synced = lg.last_synced_at[:10] if lg.last_synced_at else "Never"
                enabled_icon = "🟢" if lg.sync_enabled else "🔴"
                st.markdown(
                    f"<div class='league-card'>"
                    f"<strong>{enabled_icon} {lg.grade_name}</strong><br>"
                    f"<small>{lg.season} · Synced: {synced}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if is_admin():
                    col1, col2 = st.columns(2)
                    with col1:
                        toggle_label = "Disable" if lg.sync_enabled else "Enable"
                        if st.button(toggle_label, key=f"toggle_{lg.id}"):
                            db2 = get_db()
                            try:
                                l2 = db2.query(League).get(lg.id)
                                l2.sync_enabled = 0 if l2.sync_enabled else 1
                                db2.commit()
                            finally:
                                db2.close()
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Remove", key=f"del_{lg.id}"):
                            db2 = get_db()
                            try:
                                db2.query(League).filter_by(id=lg.id).delete()
                                db2.commit()
                            finally:
                                db2.close()
                            st.rerun()
        finally:
            db.close()

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main_app():
    sidebar_league_manager()

    st.markdown("# 🏈 League Ladders")
    st.markdown(f"Welcome back, **{st.session_state.user['username']}**! 👋")
    st.divider()

    db = get_db()
    try:
        leagues = db.query(League).order_by(League.grade_name).all()
    finally:
        db.close()

    if not leagues:
        st.info("🔗 No leagues configured yet. Ask an admin to add leagues via the sidebar.")
        return

    # Tab per league
    tab_labels = [f"🏆 {lg.grade_name}" for lg in leagues]
    tabs = st.tabs(tab_labels)

    for tab, league in zip(tabs, leagues):
        with tab:
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
                    Ladder.byes,
                    Ladder.points_for,
                    Ladder.points_against,
                    Ladder.forfeits,
                ).filter_by(grade_id=league.grade_id)\
                 .order_by(Ladder.rank).all()

                if results:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🏆 Competition", league.grade_name)
                    col2.metric("📅 Season", league.season)
                    synced = league.last_synced_at[:10] if league.last_synced_at else "Never"
                    col3.metric("🔄 Last Synced", synced)

                    st.markdown("<br>", unsafe_allow_html=True)

                    df = pd.DataFrame(results, columns=[
                        "Rank", "Team", "P", "PTS", "%", "W", "L", "D", "BYE", "F", "A", "FORF"
                    ])

                    # ── Copy to clipboard button ──────────────────────────
                    # Build tab-separated text: headers + rows
                    tsv_rows = ["\t".join(df.columns.tolist())]
                    for _, row in df.iterrows():
                        tsv_rows.append("\t".join(str(v) for v in row.tolist()))
                    tsv_data = "\\n".join(tsv_rows).replace("`", "'").replace("\\", "\\\\")

                    copy_id = f"copy_{league.grade_id}"
                    st.markdown(f"""
                    <button onclick="
                        const text = `{tsv_data}`;
                        navigator.clipboard.writeText(text.replace(/\\\\n/g, '\\n')).then(() => {{
                            const btn = document.getElementById('{copy_id}');
                            btn.innerText = '✅ Copied!';
                            btn.style.background = '#10b981';
                            setTimeout(() => {{
                                btn.innerText = '📋 Copy Table';
                                btn.style.background = '#3b82f6';
                            }}, 2000);
                        }});
                    "
                    id="{copy_id}"
                    style="
                        background:#3b82f6; color:white; border:none;
                        padding:0.45rem 1.2rem; border-radius:8px;
                        font-weight:600; font-size:0.85rem; cursor:pointer;
                        margin-bottom:0.75rem; transition:background 0.3s;
                    ">📋 Copy Table</button>
                    <span style="color:rgba(255,255,255,0.5); font-size:0.8rem; margin-left:0.5rem;">
                        Paste into Excel, Google Sheets, or anywhere
                    </span>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Team": st.column_config.TextColumn("Team", width="large"),
                            "P":    st.column_config.NumberColumn("P",    width="small"),
                            "PTS":  st.column_config.NumberColumn("PTS",  width="small"),
                            "%":    st.column_config.NumberColumn("%",    format="%.2f", width="small"),
                            "W":    st.column_config.NumberColumn("W",    width="small"),
                            "L":    st.column_config.NumberColumn("L",    width="small"),
                            "D":    st.column_config.NumberColumn("D",    width="small"),
                            "BYE":  st.column_config.NumberColumn("BYE",  width="small"),
                            "F":    st.column_config.NumberColumn("F",    width="small", help="Points For"),
                            "A":    st.column_config.NumberColumn("A",    width="small", help="Points Against"),
                            "FORF": st.column_config.NumberColumn("FORF", width="small", help="Forfeits"),
                        }
                    )
                    st.caption(f"📊 {len(df)} teams  ·  Use 📋 Copy Table to paste into Excel or Google Sheets")
                else:
                    st.info("No ladder data yet. Click **Sync All Leagues** in the sidebar.")
            finally:
                db.close()

# --------------------------------------------------
# CRON ENDPOINT — call via Render Cron Job
# --------------------------------------------------
# Render Cron Jobs run a command, not an HTTP request.
# Add this to render.yaml as a separate cron service that runs:
#   python sync_cron.py
# (see sync_cron.py below)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
