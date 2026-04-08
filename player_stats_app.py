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

load_dotenv()

st.set_page_config(page_title="Player Stats", page_icon="🏈", layout="wide")

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()

class League(Base):
    __tablename__ = 'leagues'
    id = Column(Integer, primary_key=True, autoincrement=True)
    grade_id = Column(String(255), unique=True, nullable=False)
    grade_name = Column(String(255))
    season = Column(String(255))
    url = Column(Text)
    added_by = Column(String(255))
    added_at = Column(String(255))
    last_synced_at = Column(String(255))
    sync_enabled = Column(Integer, default=1)

class PlayerGame(Base):
    __tablename__ = 'player_games'
    id = Column(Integer, primary_key=True, autoincrement=True)
    grade_id = Column(String(255), index=True)
    grade_name = Column(String(255))
    season = Column(String(255))
    round_number = Column(Integer)
    round_name = Column(String(255))
    game_id = Column(String(255), index=True)
    game_date = Column(String(255))
    team_id = Column(String(255))
    team_name = Column(String(255))
    opponent_name = Column(String(255))
    player_id = Column(String(255), index=True)
    player_name = Column(String(255))
    player_number = Column(String(50))
    goals = Column(Integer, default=0)
    behinds = Column(Integer, default=0)
    best_player_rank = Column(Integer, default=0)
    synced_at = Column(String(255))

class PlayerProfile(Base):
    __tablename__ = 'player_profiles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(255), unique=True, nullable=False)
    player_name = Column(String(255))
    fetched_at = Column(String(255))

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
    return SessionLocal()

# --------------------------------------------------
# COPY BUTTON HELPER
# --------------------------------------------------
def copy_button(df, key: str, label: str = "Copy Table"):
    """Renders a working copy button using components.html (works inside Streamlit iframe)."""
    import streamlit.components.v1 as components
    lines = ["\t".join(str(c) for c in df.columns.tolist())]
    for _, row in df.iterrows():
        lines.append("\t".join("" if v is None else str(v) for v in row.tolist()))
    tsv = "\n".join(lines).replace("&","&amp;").replace('"',"&quot;").replace("<","&lt;").replace(">","&gt;")
    safe_key = key.replace("-","_").replace(" ","_").replace("/","_").replace(".","_")
    btn_id  = f"cb_{safe_key}"
    area_id = f"ta_{safe_key}"
    # Strip emoji from label for JS safety
    safe_label = label.encode('ascii', errors='ignore').decode()
    components.html(f"""
    <textarea id="{area_id}" readonly style="position:absolute;left:-9999px;top:-9999px;opacity:0;">{tsv}</textarea>
    <button id="{btn_id}" onclick="
        var ta=document.getElementById('{area_id}');
        ta.style.position='fixed';ta.style.left='0';ta.style.top='0';ta.style.opacity='1';
        ta.select();ta.setSelectionRange(0,999999);
        var ok=document.execCommand('copy');
        ta.style.position='absolute';ta.style.left='-9999px';ta.style.opacity='0';
        var b=document.getElementById('{btn_id}');
        if(ok){{b.innerText='Copied!';b.style.background='#10b981';}}
        else{{b.innerText='Failed';b.style.background='#ef4444';}}
        setTimeout(function(){{b.innerText='{safe_label}';b.style.background='#3b82f6';}},2000);
    " style="background:#3b82f6;color:white;border:none;padding:0.42rem 1.1rem;
             border-radius:8px;font-weight:600;font-size:0.82rem;cursor:pointer;
             transition:background 0.3s;">{safe_label}
    </button>
    <span style="color:#999;font-size:0.78rem;margin-left:0.5rem;">Paste into Excel or Google Sheets with Ctrl+V</span>
    """, height=46)

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

STATS_QUERY = """
query($gradeID: ID!) {
  discoverGrade(gradeID: $gradeID) {
    rounds {
      id
      name
      number
      games {
        id
        date
        status { value }
        home { ... on DiscoverTeam { id name } }
        away { ... on DiscoverTeam { id name } }
        statistics {
          home {
            players {
              playerNumber
              player {
                ... on DiscoverParticipant { id }
                ... on DiscoverAnonymousParticipant { id name }
                ... on DiscoverRegularFillInPlayer { id name }
                ... on DiscoverGamePermitFillInPlayer { id }
                ... on DiscoverParticipantFillInPlayer { id }
              }
              statistics { count type { value label } }
            }
            bestPlayers {
              ranking
              participant {
                ... on DiscoverParticipant { id }
                ... on DiscoverAnonymousParticipant { id name }
              }
            }
          }
          away {
            players {
              playerNumber
              player {
                ... on DiscoverParticipant { id }
                ... on DiscoverAnonymousParticipant { id name }
                ... on DiscoverRegularFillInPlayer { id name }
                ... on DiscoverGamePermitFillInPlayer { id }
                ... on DiscoverParticipantFillInPlayer { id }
              }
              statistics { count type { value label } }
            }
            bestPlayers {
              ranking
              participant {
                ... on DiscoverParticipant { id }
                ... on DiscoverAnonymousParticipant { id name }
              }
            }
          }
        }
      }
    }
  }
}
"""

PROFILE_QUERY = """
query($participantID: ID!) {
  discoverParticipant(participantID: $participantID) {
    profile { firstName lastName }
  }
}
"""

def safe_post(payload):
    try:
        r = requests.post(PLAYHQ_URL, headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        return r.json().get("data")
    except Exception:
        return None

def get_player_name(player_id, name_direct, db):
    if name_direct:
        if not db.query(PlayerProfile).filter_by(player_id=player_id).first():
            db.add(PlayerProfile(player_id=player_id, player_name=name_direct,
                                 fetched_at=datetime.utcnow().isoformat()))
            db.commit()
        return name_direct
    cached = db.query(PlayerProfile).filter_by(player_id=player_id).first()
    if cached:
        return cached.player_name or f"#{player_id[:6]}"
    data = safe_post({"query": PROFILE_QUERY, "variables": {"participantID": player_id}})
    name = None
    if data and data.get("discoverParticipant"):
        profile = (data["discoverParticipant"] or {}).get("profile") or {}
        fn = profile.get("firstName", "")
        ln = profile.get("lastName", "")
        name = f"{fn} {ln}".strip() or None
    db.add(PlayerProfile(player_id=player_id, player_name=name,
                         fetched_at=datetime.utcnow().isoformat()))
    db.commit()
    return name or f"#{player_id[:6]}"

def sync_grade_stats(grade_id, grade_name, season, silent=False, resolve_names=True):
    synced_at = datetime.utcnow().isoformat()
    data = safe_post({"query": STATS_QUERY, "variables": {"gradeID": grade_id}})
    if not data:
        if not silent: st.error("❌ Failed to fetch from PlayHQ")
        return False
    rounds = data.get("discoverGrade", {}).get("rounds", [])
    db = get_db()
    added = updated = 0
    try:
        for rnd in rounds:
            round_name = rnd["name"]
            round_number = rnd.get("number", 0)
            for game in rnd.get("games", []):
                game_id = game["id"]
                game_date = game.get("date", "")
                if game.get("status", {}).get("value") not in ("FINAL", "FORFEIT"):
                    continue
                home = game.get("home", {})
                away = game.get("away", {})
                stats = game.get("statistics", {})
                for side, tid, tname, oname in [
                    ("home", home.get("id",""), home.get("name",""), away.get("name","")),
                    ("away", away.get("id",""), away.get("name",""), home.get("name","")),
                ]:
                    side_data = stats.get(side, {})
                    players = side_data.get("players", [])
                    best_players = side_data.get("bestPlayers", [])
                    bp_map = {}
                    for bp in best_players:
                        part = bp.get("participant", {}) or {}
                        pid = part.get("id")
                        if pid:
                            bp_map[pid] = (bp.get("ranking", 1), part.get("name"))
                    for pe in players:
                        player_obj = pe.get("player", {}) or {}
                        player_id = player_obj.get("id", "")
                        name_direct = player_obj.get("name")
                        player_number = pe.get("playerNumber", "")
                        if not player_id:
                            continue
                        if resolve_names:
                            player_name = get_player_name(player_id, name_direct, db)
                        else:
                            cached = db.query(PlayerProfile).filter_by(player_id=player_id).first()
                            player_name = (cached.player_name if cached else None) or name_direct or f"#{player_number}"
                        goals = behinds = 0
                        for stat in pe.get("statistics", []):
                            v = stat["type"]["value"]
                            c = stat["count"]
                            if v == "6_POINT_SCORE": goals = c
                            elif v == "1_POINT_SCORE": behinds = c
                        bp_rank = bp_map.get(player_id, (0, None))[0] if player_id in bp_map else 0
                        existing = db.query(PlayerGame).filter_by(
                            game_id=game_id, player_id=player_id, team_id=tid).first()
                        vals = dict(
                            grade_id=grade_id, grade_name=grade_name, season=season,
                            round_number=round_number, round_name=round_name,
                            game_date=game_date, team_id=tid, team_name=tname,
                            opponent_name=oname, player_name=player_name,
                            player_number=player_number, goals=goals, behinds=behinds,
                            best_player_rank=bp_rank, synced_at=synced_at)
                        if existing:
                            for k, v in vals.items(): setattr(existing, k, v)
                            updated += 1
                        else:
                            db.add(PlayerGame(player_id=player_id, **vals))
                            added += 1
        db.commit()
        if not silent:
            st.success(f"✅ {grade_name}: {added} new, {updated} updated")
        return True
    except Exception as e:
        db.rollback()
        if not silent: st.error(f"Database error: {e}")
        return False
    finally:
        db.close()

# --------------------------------------------------
# AUTH
# --------------------------------------------------
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def verify_login(username, password):
    db = get_db()
    try:
        u = db.query(User).filter_by(username=username,
                                     password_hash=hash_password(password)).first()
        return {"id": u.id, "username": u.username, "role": u.role} if u else None
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
    h1,h2,h3 { color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1e293b 0%,#334155 100%); }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stDataFrame { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .stButton>button { background: linear-gradient(135deg,#3b82f6 0%,#2563eb 100%); color:white; border:none; padding:0.75rem 2rem; border-radius:12px; font-weight:600; }
    div[data-testid="metric-container"] { background:white; padding:1.5rem; border-radius:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08); }
    .league-card { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.4rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
def login_page():
    st.markdown("<h1 style='text-align:center;'>🏈 Player Stats</h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1,2,1])
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

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
def sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['username']} ({st.session_state.user['role'].upper()})")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
        st.divider()
        if is_admin():
            st.markdown("### 🔄 Sync")
            db = get_db()
            try:
                leagues = db.query(League).filter_by(sync_enabled=1).all()
            finally:
                db.close()
            if leagues:
                if st.button("Sync All Leagues", use_container_width=True, type="primary"):
                    for lg in leagues:
                        with st.spinner(f"Syncing {lg.grade_name}..."):
                            sync_grade_stats(lg.grade_id, lg.grade_name, lg.season)
                    st.rerun()
                for lg in leagues:
                    if st.button(f"↻ {lg.grade_name[:28]}", key=f"s_{lg.id}"):
                        with st.spinner(f"Syncing {lg.grade_name}..."):
                            sync_grade_stats(lg.grade_id, lg.grade_name, lg.season)
                        st.rerun()
            else:
                st.info("No leagues configured in Ladder app.")
        st.divider()
        st.markdown("### 🏆 Leagues")
        db = get_db()
        try:
            for lg in db.query(League).order_by(League.grade_name).all():
                icon = "🟢" if lg.sync_enabled else "🔴"
                st.markdown(
                    f"<div class='league-card'><strong>{icon} {lg.grade_name}</strong><br>"
                    f"<small>{lg.season}</small></div>", unsafe_allow_html=True)
        finally:
            db.close()

# --------------------------------------------------
# SEASON STATS
# --------------------------------------------------
def season_stats_view(grade_id):
    db = get_db()
    try:
        rows = db.execute(text("""
            SELECT
                player_id,
                MAX(player_name)                                    AS player_name,
                MAX(team_name)                                      AS team_name,
                MAX(player_number)                                  AS player_number,
                COUNT(DISTINCT game_id)                             AS gp,
                COALESCE(SUM(goals), 0)                             AS g,
                COUNT(CASE WHEN best_player_rank > 0 THEN 1 END)   AS bp
            FROM player_games
            WHERE grade_id = :gid
            GROUP BY player_id
            ORDER BY g DESC, bp DESC, gp DESC
        """), {"gid": grade_id}).fetchall()

        if not rows:
            st.info("No stats synced yet. Click **Sync All Leagues** in the sidebar.")
            return

        df = pd.DataFrame(rows, columns=["player_id","Player","Team","#","GP","G","BP"])
        df = df.drop(columns=["player_id"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Players", len(df))
        c2.metric("🏈 Teams", df["Team"].nunique())
        top_g = df.sort_values("G", ascending=False).iloc[0]
        c3.metric("🥇 Top Scorer", f"{top_g['Player']} ({int(top_g['G'])}G)")
        top_bp = df.sort_values("BP", ascending=False).iloc[0]
        c4.metric("⭐ Most BPs", f"{top_bp['Player']} ({int(top_bp['BP'])})")

        st.markdown("<br>", unsafe_allow_html=True)

        fc1, fc2 = st.columns([3, 2])
        with fc1:
            teams = ["All Teams"] + sorted(df["Team"].dropna().unique().tolist())
            sel_team = st.selectbox("Filter by Team", teams, key=f"tf_{grade_id}")
        with fc2:
            sort_by = st.selectbox("Sort by", ["G", "BP", "GP"], key=f"sb_{grade_id}")

        filtered = df.copy()
        if sel_team != "All Teams":
            filtered = filtered[filtered["Team"] == sel_team]
        filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)
        filtered.index += 1

        # Export df includes rank number
        export_df = filtered[["Player","Team","#","GP","G","BP"]].copy()
        export_df.insert(0, "Rank", range(1, len(export_df)+1))
        copy_button(export_df, key=f"season_{grade_id}")

        st.dataframe(
            filtered[["Player","Team","#","GP","G","BP"]],
            use_container_width=True,
            hide_index=False,
            height=600,
            column_config={
                "Player": st.column_config.TextColumn("Player", width="large"),
                "Team":   st.column_config.TextColumn("Team",   width="medium"),
                "#":      st.column_config.TextColumn("#",      width="small"),
                "GP":     st.column_config.NumberColumn("GP",   width="small", help="Games Played"),
                "G":      st.column_config.NumberColumn("G",    width="small", help="Total Goals"),
                "BP":     st.column_config.NumberColumn("BP",   width="small", help="Best Player Awards"),
            }
        )
        st.caption("GP = Games Played · G = Goals · BP = Best Player Awards")

    finally:
        db.close()

# --------------------------------------------------
# ROUND STATS
# --------------------------------------------------
def round_stats_view(grade_id):
    db = get_db()
    try:
        rounds = db.execute(text("""
            SELECT DISTINCT round_number, round_name FROM player_games
            WHERE grade_id = :gid ORDER BY round_number
        """), {"gid": grade_id}).fetchall()

        if not rounds:
            st.info("No stats available yet.")
            return

        options = {r[1]: r[0] for r in rounds}
        sel = st.selectbox("Select Round", list(options.keys()), key=f"rsel_{grade_id}")
        rnum = options[sel]

        rows = db.execute(text("""
            SELECT player_name, team_name, opponent_name, player_number,
                   goals, behinds, best_player_rank
            FROM player_games
            WHERE grade_id = :gid AND round_number = :rn
            ORDER BY team_name, goals DESC
        """), {"gid": grade_id, "rn": rnum}).fetchall()

        if not rows:
            st.info("No stats for this round.")
            return

        df = pd.DataFrame(rows, columns=["Player","Team","vs","#","G","B","BP Rank"])
        df["BP"] = (df["BP Rank"] > 0).astype(int)
        df = df.drop(columns=["BP Rank"])

        # One button copies everything for the entire round
        full_export = df[["Team","Player","#","vs","G","B","BP"]].copy()
        copy_button(full_export, key=f"round_full_{grade_id}_{rnum}",
                    label="📋 Copy Full Round (All Teams)")

        st.markdown("<br>", unsafe_allow_html=True)

        for team in sorted(df["Team"].unique()):
            tdf = df[df["Team"] == team][["Player","#","vs","G","B","BP"]].copy()
            tdf = tdf.sort_values("G", ascending=False).reset_index(drop=True)
            tdf.index += 1
            total_g = int(tdf["G"].sum())
            total_bp = int(tdf["BP"].sum())
            safe_team = team.replace(" ", "_").replace("/", "_")
            with st.expander(f"🏈 {team}  ·  {total_g}G  ·  {total_bp} BP", expanded=True):
                copy_button(tdf, key=f"round_{grade_id}_{rnum}_{safe_team}",
                            label=f"📋 Copy {team}")
                st.dataframe(tdf, use_container_width=True, hide_index=False,
                    column_config={
                        "Player": st.column_config.TextColumn("Player", width="large"),
                        "#":  st.column_config.TextColumn("#",  width="small"),
                        "vs": st.column_config.TextColumn("vs", width="medium"),
                        "G":  st.column_config.NumberColumn("G",  width="small"),
                        "B":  st.column_config.NumberColumn("B",  width="small"),
                        "BP": st.column_config.NumberColumn("BP", width="small"),
                    })
    finally:
        db.close()

# --------------------------------------------------
# LEADERBOARD
# --------------------------------------------------
def leaderboard_view(grade_id):
    db = get_db()
    try:
        rows = db.execute(text("""
            SELECT player_name, team_name,
                   COUNT(DISTINCT game_id) AS gp,
                   COALESCE(SUM(goals),0) AS g,
                   COUNT(CASE WHEN best_player_rank > 0 THEN 1 END) AS bp
            FROM player_games WHERE grade_id = :gid
            GROUP BY player_id, player_name, team_name
        """), {"gid": grade_id}).fetchall()

        if not rows:
            st.info("No data yet.")
            return

        df = pd.DataFrame(rows, columns=["Player","Team","GP","G","BP"])

        t1, t2 = st.tabs(["🥇 Top Goal Kickers", "⭐ Top Best Players"])

        with t1:
            top = df.nlargest(15,"G").reset_index(drop=True)
            top.index += 1
            top["Avg G"] = (top["G"] / top["GP"].replace(0,1)).round(2)
            export = top[["Player","Team","GP","G","Avg G"]].copy()
            export.insert(0, "Rank", range(1, len(export)+1))
            copy_button(export, key=f"lb_goals_{grade_id}", label="📋 Copy Goal Kickers")
            st.dataframe(top[["Player","Team","GP","G","Avg G"]], use_container_width=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", width="large"),
                    "Team":   st.column_config.TextColumn("Team",   width="medium"),
                    "GP":     st.column_config.NumberColumn("GP",   width="small"),
                    "G":      st.column_config.NumberColumn("G",    width="small"),
                    "Avg G":  st.column_config.NumberColumn("Avg G/Game", format="%.2f", width="small"),
                })

        with t2:
            top_bp = df[df["BP"]>0].nlargest(15,"BP").reset_index(drop=True)
            top_bp.index += 1
            export_bp = top_bp[["Player","Team","GP","BP","G"]].copy()
            export_bp.insert(0, "Rank", range(1, len(export_bp)+1))
            copy_button(export_bp, key=f"lb_bp_{grade_id}", label="📋 Copy Best Players")
            st.dataframe(top_bp[["Player","Team","GP","BP","G"]], use_container_width=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", width="large"),
                    "Team":   st.column_config.TextColumn("Team",   width="medium"),
                    "GP":     st.column_config.NumberColumn("GP",   width="small"),
                    "BP":     st.column_config.NumberColumn("BP",   width="small"),
                    "G":      st.column_config.NumberColumn("G",    width="small"),
                })
    finally:
        db.close()

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main_app():
    sidebar()
    st.markdown("# 🏈 Player Statistics")
    st.markdown(f"Welcome back, **{st.session_state.user['username']}**! 👋")
    st.divider()

    db = get_db()
    try:
        leagues = db.query(League).filter_by(sync_enabled=1).order_by(League.grade_name).all()
    finally:
        db.close()

    if not leagues:
        st.info("No leagues configured. Add them in the Ladder app first.")
        return

    league_tabs = st.tabs([f"🏆 {lg.grade_name}" for lg in leagues])
    for tab, league in zip(league_tabs, leagues):
        with tab:
            s1, s2, s3 = st.tabs(["📊 Season Stats", "📅 Round Stats", "🏅 Leaderboard"])
            with s1: season_stats_view(league.grade_id)
            with s2: round_stats_view(league.grade_id)
            with s3: leaderboard_view(league.grade_id)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
