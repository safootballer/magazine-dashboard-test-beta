import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

# MUST BE FIRST
st.set_page_config(page_title="PlayHQ Debug", layout="wide")

st.title("🧪 PlayHQ Debug App")

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
    result {
      home { score }
      away { score }
    }
  }
}
"""

def extract_match_id(url):
    return url.rstrip("/").split("/")[-1]

st.subheader("1️⃣ Paste a PlayHQ match URL")

url = st.text_input(
    "Match URL",
    value="https://www.playhq.com/afl/org/adelaide-footy-league/adelaide-footy-league-2025/mens-division-1-btr-excavations/game-centre/0f653565"
)

if st.button("🔍 Fetch Match"):
    try:
        match_id = extract_match_id(url)

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

        st.write("HTTP Status:", r.status_code)

        r.raise_for_status()

        data = r.json()
        st.success("✅ Data fetched")

        st.json(data)

    except Exception as e:
        st.error("❌ Error occurred")
        st.exception(e)
