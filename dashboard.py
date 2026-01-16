import streamlit as st
import requests
import json

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Magazine Automation",
    layout="wide"
)

st.title("🏉 Magazine Automation")

# =========================
# FETCH MATCH
# =========================
st.header("Fetch Match from Website")

url = st.text_input(
    "Paste match URL",
    placeholder="https://www.playhq.com/..."
)

fetch_clicked = st.button("Fetch & Save")

snapshot_id = None

if fetch_clicked:
    if not url.strip():
        st.warning("Please paste a match URL")
    else:
        with st.spinner("Fetching match and creating snapshot..."):
            r = requests.post(
                f"{API}/ingest",
                json={"url": url},
                timeout=60
            )

        if r.status_code == 200:
            data = r.json()
            snapshot_id = data["snapshot_id"]

            if data.get("existing"):
                st.info("Match already fetched. Loaded existing snapshot.")
            else:
                st.success("New snapshot saved successfully.")
        else:
            st.error(r.text)

# =========================
# LOAD SNAPSHOTS
# =========================
st.divider()
st.header("Saved Matches")

snaps = requests.get(f"{API}/snapshots").json()

if not snaps:
    st.info("No matches fetched yet.")
    st.stop()

snapshot_map = {
    f"Snapshot {s['id']} (created {s['created_at']})": s["id"]
    for s in snaps
}

# Auto-select latest or fetched snapshot
default_index = 0
if snapshot_id:
    for i, v in enumerate(snapshot_map.values()):
        if v == snapshot_id:
            default_index = i
            break

selected_label = st.selectbox(
    "Select a snapshot",
    list(snapshot_map.keys()),
    index=default_index
)

selected_snapshot_id = snapshot_map[selected_label]

# =========================
# SNAPSHOT PREVIEW
# =========================
snap = requests.get(
    f"{API}/snapshots/{selected_snapshot_id}"
).json()

snapshot = json.loads(snap["snapshot_json"])

st.divider()
st.header("Snapshot Preview")

col1, col2 = st.columns(2)

# -------- LEFT: Structured Snapshot
with col1:
    st.subheader("Structured Snapshot (What the editor sees)")

    for section, content in snapshot.items():
        if section == "raw_text":
            continue

        if not content:
            continue

        st.markdown(f"### {section.replace('_', ' ').title()}")
        st.text("\n".join(content))

# -------- RIGHT: ChatGPT Input
with col2:
    st.subheader("Snapshot Text (ChatGPT Input)")
    st.text_area(
        "This exact text will be sent to ChatGPT",
        snap["snapshot_text"],
        height=600
    )

# =========================
# NEXT ACTIONS (PLACEHOLDERS)
# =========================
st.divider()
st.subheader("Next Actions")

st.button("📰 Generate Magazine Article (coming next)")
st.button("🌐 Generate Web Article (coming next)")
st.button("📱 Generate Social Captions (coming next)")
