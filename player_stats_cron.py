"""
player_stats_cron.py
--------------------
Daily cron job to sync player stats for all enabled leagues.

Render Cron Job setup:
  Command:  python player_stats_cron.py
  Schedule: 30 3 * * *   (3:30am UTC daily — runs after ladder cron)
  Env vars: DATABASE_URL (same as web service)
"""

import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///magazine.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

PLAYHQ_URL = "https://api.playhq.com/graphql"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Origin": "https://www.playhq.com",
    "Referer": "https://www.playhq.com/",
    "tenant": "afl"
}

PLAYER_STATS_QUERY = """
query($gradeID: ID!) {
  discoverGrade(gradeID: $gradeID) {
    rounds {
      id
      name
      number
      games {
        id
        date
        status { name value }
        home {
          ... on DiscoverTeam { id name }
        }
        away {
          ... on DiscoverTeam { id name }
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
                ... on DiscoverParticipant { id }
                ... on DiscoverAnonymousParticipant { id name }
                ... on DiscoverRegularFillInPlayer { id name }
                ... on DiscoverGamePermitFillInPlayer { id }
                ... on DiscoverParticipantFillInPlayer { id }
              }
              statistics {
                count
                type { value label }
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
              statistics {
                count
                type { value label }
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
    profile {
      firstName
      lastName
    }
  }
}
"""

def log(msg):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}", flush=True)

def safe_post(payload):
    try:
        r = requests.post(PLAYHQ_URL, headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        return r.json().get("data")
    except Exception as e:
        log(f"  Request error: {e}")
        return None

def fetch_player_name(db, player_id):
    try:
        cached = db.execute(
            text("SELECT player_name FROM player_profiles WHERE player_id = :pid"),
            {"pid": player_id}
        ).fetchone()
        if cached:
            return cached[0] or f"#{player_id[:8]}"
    except Exception:
        pass

    data = safe_post({"query": PROFILE_QUERY, "variables": {"participantID": player_id}})
    name = None
    if data and data.get("discoverParticipant"):
        profile = data["discoverParticipant"].get("profile") or {}
        fn = profile.get("firstName", "")
        ln = profile.get("lastName", "")
        name = f"{fn} {ln}".strip() or None

    try:
        db.execute(text("""
            INSERT INTO player_profiles (player_id, player_name, fetched_at)
            VALUES (:pid, :name, :fat)
            ON CONFLICT (player_id) DO UPDATE SET player_name=:name, fetched_at=:fat
        """), {"pid": player_id, "name": name, "fat": datetime.utcnow().isoformat()})
        db.commit()
    except Exception:
        db.rollback()

    return name or f"#{player_id[:8]}"

def sync_grade(db, grade_id, grade_name, season):
    log(f"  Fetching stats for {grade_name}...")
    synced_at = datetime.utcnow().isoformat()

    data = safe_post({"query": PLAYER_STATS_QUERY, "variables": {"gradeID": grade_id}})
    if not data:
        log(f"  ❌ No data returned")
        return False

    rounds = data.get("discoverGrade", {}).get("rounds", [])
    added = updated = 0

    for rnd in rounds:
        round_name = rnd["name"]
        round_number = rnd.get("number", 0)

        for game in rnd.get("games", []):
            game_id = game["id"]
            game_date = game.get("date", "")
            status = game.get("status", {}).get("value", "")

            if status not in ("FINAL", "FORFEIT"):
                continue

            home = game.get("home", {})
            away = game.get("away", {})
            home_id = home.get("id", "")
            home_name = home.get("name", "")
            away_id = away.get("id", "")
            away_name = away.get("name", "")

            stats_data = game.get("statistics", {})

            for side, tid, tname, oid, oname in [
                ("home", home_id, home_name, away_id, away_name),
                ("away", away_id, away_name, home_id, home_name),
            ]:
                players = stats_data.get(side, {}).get("players", [])

                for pe in players:
                    player_obj = pe.get("player", {})
                    player_id = player_obj.get("id", "")
                    player_name_direct = player_obj.get("name")
                    player_number = pe.get("playerNumber", "")
                    player_stats = pe.get("statistics", [])

                    if not player_id:
                        continue

                    if player_name_direct:
                        player_name = player_name_direct
                    else:
                        player_name = fetch_player_name(db, player_id)

                    if not player_stats:
                        player_stats = [{"count": 1, "type": {"value": "APPEARANCE", "label": "Appearance"}}]

                    for stat in player_stats:
                        stat_type = stat["type"]["value"]
                        stat_label = stat["type"]["label"]
                        stat_count = stat["count"]

                        existing = db.execute(text("""
                            SELECT id FROM player_stats
                            WHERE game_id=:gid AND player_id=:pid AND team_id=:tid AND stat_type=:st
                        """), {"gid": game_id, "pid": player_id, "tid": tid, "st": stat_type}).fetchone()

                        if existing:
                            db.execute(text("""
                                UPDATE player_stats SET
                                    player_name=:pname, stat_count=:sc, synced_at=:sat
                                WHERE id=:id
                            """), {"pname": player_name, "sc": stat_count, "sat": synced_at, "id": existing[0]})
                            updated += 1
                        else:
                            db.execute(text("""
                                INSERT INTO player_stats
                                    (grade_id, grade_name, season, round_number, round_name,
                                     game_id, game_date, team_id, team_name, opponent_id, opponent_name,
                                     player_id, player_name, player_number, stat_type, stat_label,
                                     stat_count, synced_at)
                                VALUES
                                    (:gid, :gname, :season, :rnum, :rname,
                                     :gameid, :gdate, :tid, :tname, :oid, :oname,
                                     :pid, :pname, :pnum, :st, :sl, :sc, :sat)
                            """), {
                                "gid": grade_id, "gname": grade_name, "season": season,
                                "rnum": round_number, "rname": round_name,
                                "gameid": game_id, "gdate": game_date,
                                "tid": tid, "tname": tname, "oid": oid, "oname": oname,
                                "pid": player_id, "pname": player_name, "pnum": player_number,
                                "st": stat_type, "sl": stat_label, "sc": stat_count, "sat": synced_at
                            })
                            added += 1

    db.commit()
    log(f"  ✅ {grade_name}: {added} added, {updated} updated")
    return True

def main():
    log("=== Player stats sync started ===")
    db = SessionLocal()
    try:
        leagues = db.execute(
            text("SELECT grade_id, grade_name, season FROM leagues WHERE sync_enabled = 1")
        ).fetchall()

        if not leagues:
            log("No leagues configured.")
            return

        log(f"Syncing {len(leagues)} leagues...")
        ok = fail = 0
        for grade_id, grade_name, season in leagues:
            log(f"Processing: {grade_name}")
            try:
                if sync_grade(db, grade_id, grade_name, season):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                log(f"  ❌ Error: {e}")
                db.rollback()
                fail += 1

        log(f"=== Done: {ok} succeeded, {fail} failed ===")
    finally:
        db.close()

if __name__ == "__main__":
    main()
