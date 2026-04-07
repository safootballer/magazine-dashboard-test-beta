import streamlit as st
import os
import base64
from pathlib import Path

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="The South Australian Footballer - Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* ── Background: deep forest green → dark teal (distinct from navy) ── */
    .stApp {
        background: linear-gradient(160deg, #0a1f0a 0%, #0d2b1a 35%, #0a2a2a 70%, #062020 100%);
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        padding: 1.5rem 2rem;
        background: rgba(0,0,0,0.5);
        border-radius: 14px;
        border: 1px solid rgba(44,163,238,0.25);
    }
    .stat-item { text-align: center; }
    .stat-number {
        font-size: 2.4rem;
        font-weight: 900;
        color: #e6fe00;
        text-shadow: 0 2px 8px rgba(230,254,0,0.3);
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.45);
        margin-top: 0.25rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Service cards ── */
    .service-card {
        background: rgba(0,0,0,0.7);
        border-radius: 18px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        border: 2px solid rgba(44,163,238,0.25);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        min-height: 360px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        animation: fadeInUp 0.6s ease-out;
    }
    .service-card:hover {
        transform: translateY(-6px);
        border-color: #2ca3ee;
        box-shadow: 0 20px 60px rgba(44,163,238,0.3);
    }
    .service-card.safire-card:hover {
        border-color: #e6fe00;
        box-shadow: 0 20px 60px rgba(230,254,0,0.2);
    }

    .service-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .service-logo {
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin: 0 auto 1rem auto;
        display: block;
        filter: drop-shadow(0 4px 12px rgba(44,163,238,0.4));
    }
    .service-title {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }
    .service-title.safire-title { color: #2ca3ee; }
    .service-description {
        color: rgba(255,255,255,0.6);
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
        min-height: 55px;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-safire  { background: #e6fe00; color: #000000 !important; }
    .badge-admin   { background: linear-gradient(135deg, #2ca3ee 0%, #00b8f1 100%); }
    .badge-stats   { background: transparent; border: 2px solid #e6fe00; color: #e6fe00 !important; }

    /* ── Launch buttons (HTML only — used for SAFie card wrapper) ── */
    .launch-btn {
        display: inline-block;
        width: 100%;
        padding: 0.85rem 1.5rem;
        border-radius: 50px;
        font-weight: 800;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-sizing: border-box;
        text-decoration: none !important;
        cursor: pointer;
    }
    .launch-btn-safire {
        background: #e6fe00;
        color: #000000 !important;
        box-shadow: 0 6px 24px rgba(230,254,0,0.3);
    }
    .launch-btn-safire:hover {
        background: #d4eb00;
        box-shadow: 0 10px 36px rgba(230,254,0,0.5);
        color: #000000 !important;
    }

    /* ── Quick-access links ── */
    .quick-link {
        color: #2ca3ee !important;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .quick-link:hover { color: #00b8f1 !important; text-decoration: underline !important; }

    hr { border-color: rgba(44,163,238,0.25) !important; }

    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.85rem;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(44,163,238,0.2);
    }
    .footer strong { color: #2ca3ee; }

    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(30px) }
        to   { opacity:1; transform:translateY(0) }
    }

    /* ── Override Streamlit button styles for card buttons ── */
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.7rem 1.2rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    /* Blue buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2ca3ee 0%, #00b8f1 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(44,163,238,0.35) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        box-shadow: 0 10px 32px rgba(44,163,238,0.55) !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Helper: load image as base64
# --------------------------------------------------
def img_to_b64(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = p.suffix.lstrip(".")
    return f"data:image/{ext};base64,{b64}"


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # ── Header ───────────────────────────────────────────────────
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        try:
            st.image("assets/logo2.png", width=130)
        except Exception:
            st.markdown("<p style='color:#2ca3ee;text-align:center;font-weight:700;'>SAFie</p>", unsafe_allow_html=True)
    with col_m:
        st.markdown("""
        <div style="text-align:center; padding-top:10px;">
            <h1 style="margin:0; font-size:2.4rem; font-weight:900; color:#2ca3ee; letter-spacing:3px;">
                SOUTH AUSTRALIAN FOOTBALLER
            </h1>
            <span style="display:inline-block; background:#e6fe00; color:#000; font-size:0.72rem;
                         font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                         border-radius:20px; padding:3px 16px; margin-top:8px;">
                Master Dashboard
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        try:
            st.image("assets/logo.png", width=130)
        except Exception:
            st.markdown("<p style='color:white;text-align:center;font-weight:700;'>SA Footballer</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats bar ────────────────────────────────────────────────
    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-number">4</div>
            <div class="stat-label">Active Services</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Availability</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">⚡</div>
            <div class="stat-label">AI Powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Service Cards ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    # ── SAFie card (HTML link wrapper — works fine) ──────────────
    with col1:
        safire_logo = img_to_b64("assets/logo2.png")
        logo_html = (f'<img src="{safire_logo}" class="service-logo" alt="SAFie" />'
                     if safire_logo else '<div class="service-icon">📰</div>')
        st.markdown(f"""
        <a href="https://magazine-user.onrender.com" target="_blank" rel="noopener noreferrer"
           style="text-decoration:none; display:block;">
            <div class="service-card safire-card">
                <div>
                    {logo_html}
                    <div class="service-title safire-title">SAFie</div>
                    <div style="color:#e6fe00; font-size:0.72rem; font-weight:700;
                                letter-spacing:0.1em; text-transform:uppercase;
                                margin-top:-4px; margin-bottom:10px;">AI by SA Footballer</div>
                    <div class="service-description">
                        AI-powered match report generation for magazines, web articles,
                        and social media posts in seconds.
                    </div>
                    <span class="badge badge-safire">AI · Core</span>
                </div>
                <div>
                    <span class="launch-btn launch-btn-safire">🚀 Launch SAFie</span>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)

    # ── Admin card (Streamlit button — bypasses HTML link issue) ──
    with col2:
        st.markdown("""
        <div class="service-card" style="min-height:360px;">
            <div>
                <div class="service-icon">⚙️</div>
                <div class="service-title">Admin Dashboard</div>
                <div class="service-description">
                    Complete analytics, user management, cost tracking,
                    match links, and full system administration.
                </div>
                <span class="badge badge-admin">Admin</span>
            </div>
            <div id="admin-btn-placeholder"></div>
        </div>
        """, unsafe_allow_html=True)
        # Real Streamlit button rendered below the card HTML
        if st.button("🚀 Launch Admin Dashboard", key="btn_admin", type="primary", use_container_width=True):
            st.markdown('<script>window.open("https://magazine-admin.onrender.com","_blank");</script>',
                        unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align:center; margin-top:6px;">'
            '<a href="https://magazine-admin.onrender.com" target="_blank" '
            'style="color:#2ca3ee; font-size:0.82rem; font-weight:600;">↗ Open in new tab</a></p>',
            unsafe_allow_html=True
        )

    # ── League & Ladder card (TWO buttons) ───────────────────────
    with col3:
        st.markdown("""
        <div class="service-card" style="min-height:360px;">
            <div>
                <div class="service-icon">🏆</div>
                <div class="service-title">League &amp; Ladder</div>
                <div class="service-description">
                    Live league ladders, standings, and player statistics
                    synced daily from PlayHQ.
                </div>
                <span class="badge badge-stats">Stats</span>
            </div>
            <div id="league-btn-placeholder"></div>
        </div>
        """, unsafe_allow_html=True)

        # Two side-by-side buttons
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🏅 Ladder", key="btn_ladder", type="primary", use_container_width=True):
                st.markdown('<script>window.open("https://ladder-hu0e.onrender.com","_blank");</script>',
                            unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center; margin-top:4px;">'
                '<a href="https://ladder-hu0e.onrender.com" target="_blank" '
                'style="color:#2ca3ee; font-size:0.78rem; font-weight:600;">↗ Open</a></p>',
                unsafe_allow_html=True
            )
        with b2:
            if st.button("📊 Player Stats", key="btn_stats", type="primary", use_container_width=True):
                st.markdown('<script>window.open("https://player-stats-app-a176.onrender.com","_blank");</script>',
                            unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center; margin-top:4px;">'
                '<a href="https://player-stats-app-a176.onrender.com" target="_blank" '
                'style="color:#2ca3ee; font-size:0.78rem; font-weight:600;">↗ Open</a></p>',
                unsafe_allow_html=True
            )

    # ── Quick-access links ────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "<h3 style='color:#2ca3ee; border-bottom:3px solid #2ca3ee; "
        "padding-bottom:6px; display:inline-block;'>🔗 Quick Access Links</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    links = [
        ("📰", "SAFie", "#2ca3ee", "https://magazine-user.onrender.com",
         "AI-powered match report generation for magazines, web articles…"),
        ("⚙️", "Admin Dashboard", "#ffffff", "https://magazine-admin.onrender.com",
         "Complete analytics, user management, cost tracking…"),
        ("🏅", "Ladder", "#ffffff", "https://ladder-hu0e.onrender.com",
         "Live league ladders and standings from PlayHQ…"),
        ("📊", "Player Stats", "#ffffff", "https://player-stats-app-a176.onrender.com",
         "GP, Goals, Best Player awards per league…"),
    ]

    qa, qb, qc, qd = st.columns(4)
    for col, (icon, name, colour, url, desc) in zip([qa, qb, qc, qd], links):
        with col:
            st.markdown(f"""
            <p style='color:{colour}; font-weight:800; font-size:1rem; margin-bottom:4px;'>
                {icon} {name}
            </p>
            <a href="{url}" target="_blank" rel="noopener noreferrer" class="quick-link">
                Open in New Tab →
            </a>
            <p style='color:rgba(255,255,255,0.45); font-size:0.82rem; margin-top:0.4rem;'>
                {desc}
            </p>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <p>🏈 <strong>The South Australian Footballer</strong> · Professional Sports Content Management</p>
        <p style="margin-top:0.4rem;">
            Powered by <span style="color:#2ca3ee; font-weight:700;">SAFie AI</span>
            &nbsp;·&nbsp;
            <span style="color:#e6fe00; font-weight:600;">AI by SA Footballer</span>
            &nbsp;·&nbsp; Built with Streamlit · Deployed on Render
        </p>
        <p style="margin-top:0.4rem; font-size:0.78rem; color:rgba(255,255,255,0.25);">
            Developed by Mian Talha Sarfraz &nbsp;·&nbsp;
            <a href="https://github.com/talha-11-11" style="color:#2ca3ee;">GitHub</a> &nbsp;·&nbsp;
            <a href="mailto:talhasarfraz29@gmail.com" style="color:#2ca3ee;">talhasarfraz29@gmail.com</a> &nbsp;·&nbsp;
            <a href="https://www.upwork.com/freelancers/~0128359f0564f06967" style="color:#2ca3ee;">Upwork</a>
        </p>
        <p style="margin-top:0.8rem; font-size:0.75rem; color:rgba(255,255,255,0.2);">
            © 2026 The South Australian Footballer. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
