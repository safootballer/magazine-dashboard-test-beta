import requests
import json

# --------------------------------------
# CONFIG
# --------------------------------------
PLAYHQ_GRAPHQL_URL = "https://api.playhq.com/graphql"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Origin": "https://www.playhq.com",
    "Referer": "https://www.playhq.com/",
    "tenant": "afl",
}

TEST_MATCH_URL = "https://www.playhq.com/afl/.../game-centre/0f653565"

# --------------------------------------
# HELPERS
# --------------------------------------
def extract_match_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


# --------------------------------------
# GRAPHQL QUERY (SAFE, MINIMAL)
# --------------------------------------
GRAPHQL_QUERY = """
query gameView($gameId: ID!) {
  discoverGame(gameID: $gameId) {
    id
    date
    home { ... on DiscoverTeam { name } }
    away { ... on DiscoverTeam { name } }

    statistics {
      home {
        players {
          playerNumber
          player {
            __typename
            ... on DiscoverParticipant {
              profile { firstName lastName }
            }
            ... on DiscoverParticipantFillInPlayer {
              profile { firstName lastName }
            }
            ... on DiscoverGamePermitFillInPlayer {
              profile { firstName lastName }
            }
          }
        }
        periods {
          period { value }
          statistics {
            type { value }
            count
          }
        }
      }

      away {
        players {
          playerNumber
          player {
            __typename
            ... on DiscoverParticipant {
              profile { firstName lastName }
            }
            ... on DiscoverParticipantFillInPlayer {
              profile { firstName lastName }
            }
            ... on DiscoverGamePermitFillInPlayer {
              profile { firstName lastName }
            }
          }
        }
        periods {
          period { value }
          statistics {
            type { value }
            count
          }
        }
      }
    }
  }
}
"""

# --------------------------------------
# MAIN
# --------------------------------------
if __name__ == "__main__":
    match_id = extract_match_id(TEST_MATCH_URL)

    payload = {
        "operationName": "gameView",
        "variables": {"gameId": match_id},
        "query": GRAPHQL_QUERY,
    }

    print("📡 Fetching PlayHQ data...\n")

    r = requests.post(
        PLAYHQ_GRAPHQL_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )

    response = r.json()

    if "errors" in response:
        print("❌ GraphQL ERROR:")
        print(json.dumps(response, indent=2))
        exit(1)

    print("✅ SUCCESS — RAW PLAYHQ RESPONSE:\n")
    print(json.dumps(response, indent=2))

    # --------------------------------------
    # QUICK STRUCTURE CHECKS
    # --------------------------------------
    game = response["data"]["discoverGame"]

    print("\n🔎 QUICK CHECKS")
    print("- Home team:", game["home"]["name"])
    print("- Away team:", game["away"]["name"])
    print("- Home players count:", len(game["statistics"]["home"]["players"]))
    print("- Away players count:", len(game["statistics"]["away"]["players"]))
    print("- Home periods:", [p["period"]["value"] for p in game["statistics"]["home"]["periods"]])
