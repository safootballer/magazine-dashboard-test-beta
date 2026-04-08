"""
sync_cron.py
------------
Run this as a Render Cron Job every day to auto-sync all saved leagues.

Render setup:
  1. In your Render dashboard, go to "New" → "Cron Job"
  2. Connect the same repo as your Streamlit app
  3. Set the command to:  python sync_cron.py
  4. Set schedule to:     0 3 * * *   (3am UTC daily)
  5. Add the same DATABASE_URL environment variable as your web service

That's it — no extra API key needed.
"""

import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
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

def sync_one(db, grade_id, grade_name, season):
    synced_at = datetime.utcnow().isoformat()
    data = safe_post({"query": LADDER_QUERY, "variables": {"gradeID": grade_id}})
    if not data:
        log(f"  ❌ No data returned for {grade_name}")
        return False

    grade = data.get("discoverGrade")
    if not grade or not grade.get("ladder"):
        log(f"  ❌ Empty ladder for {grade_name}")
        return False

    rows_updated = 0
    rows_added = 0

    for block in grade["ladder"]:
        round_id = block["generatedFrom"]["id"]
        round_name = block["generatedFrom"]["name"]

        for idx, row in enumerate(block["standings"], start=1):
            existing = db.execute(
                text("""
                    SELECT id FROM ladder
                    WHERE team_id = :tid AND season = :s AND round_id = :rid
                """),
                {"tid": row["team"]["id"], "s": season, "rid": round_id}
            ).fetchone()

            if existing:
                db.execute(
                    text("""
                        UPDATE ladder SET
                            grade_id=:gid, grade_name=:gname, round_name=:rname,
                            team_name=:tname, rank=:rank, played=:played,
                            wins=:wins, losses=:losses, draws=:draws, byes=:byes,
                            points=:pts, percentage=:pct,
                            points_for=:pf, points_against=:pa, forfeits=:forf,
                            synced_at=:sat
                        WHERE id=:id
                    """),
                    {
                        "gid": grade_id, "gname": grade_name, "rname": round_name,
                        "tname": row["team"]["name"], "rank": idx,
                        "played": row["played"], "wins": row["won"],
                        "losses": row["lost"], "draws": row["drawn"],
                        "byes": row.get("byes", 0), "pts": row["competitionPoints"],
                        "pct": row["alternatePercentage"],
                        "pf": row.get("pointsFor", 0), "pa": row.get("pointsAgainst", 0),
                        "forf": row.get("forfeits", 0),
                        "sat": synced_at, "id": existing[0]
                    }
                )
                rows_updated += 1
            else:
                db.execute(
                    text("""
                        INSERT INTO ladder
                            (grade_id, grade_name, season, round_id, round_name,
                             team_id, team_name, rank, played, wins, losses,
                             draws, byes, points, percentage,
                             points_for, points_against, forfeits, synced_at)
                        VALUES
                            (:gid, :gname, :season, :rid, :rname,
                             :tid, :tname, :rank, :played, :wins, :losses,
                             :draws, :byes, :pts, :pct,
                             :pf, :pa, :forf, :sat)
                    """),
                    {
                        "gid": grade_id, "gname": grade_name, "season": season,
                        "rid": round_id, "rname": round_name,
                        "tid": row["team"]["id"], "tname": row["team"]["name"],
                        "rank": idx, "played": row["played"], "wins": row["won"],
                        "losses": row["lost"], "draws": row["drawn"],
                        "byes": row.get("byes", 0), "pts": row["competitionPoints"],
                        "pct": row["alternatePercentage"],
                        "pf": row.get("pointsFor", 0), "pa": row.get("pointsAgainst", 0),
                        "forf": row.get("forfeits", 0), "sat": synced_at
                    }
                )
                rows_added += 1

    # Update leagues table
    db.execute(
        text("UPDATE leagues SET last_synced_at=:sat WHERE grade_id=:gid"),
        {"sat": synced_at, "gid": grade_id}
    )

    log(f"  ✅ {grade_name}: {rows_updated} updated, {rows_added} added")
    return True


def main():
    log("=== Auto-sync started ===")

    db = SessionLocal()
    try:
        leagues = db.execute(
            text("SELECT grade_id, grade_name, season FROM leagues WHERE sync_enabled = 1")
        ).fetchall()

        if not leagues:
            log("No leagues with sync enabled. Nothing to do.")
            return

        log(f"Found {len(leagues)} leagues to sync")

        success = 0
        fail = 0
        for grade_id, grade_name, season in leagues:
            log(f"Syncing: {grade_name} ({grade_id})")
            ok = sync_one(db, grade_id, grade_name, season)
            db.commit()
            if ok:
                success += 1
            else:
                fail += 1

        log(f"=== Done: {success} succeeded, {fail} failed ===")

    except Exception as e:
        db.rollback()
        log(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
