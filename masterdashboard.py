import streamlit as st
import os
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
# Custom CSS  — Brand colours: #2ca3ee / #00b8f1 / #e6fe00 / #000000 / #ffffff
# --------------------------------------------------
st.markdown("""
<style>
    /* ── Background: black → dark navy (matches SAFie app) ── */
    .stApp {
        background: linear-gradient(160deg, #000000 0%, #0a1a2e 35%, #0d2b4e 70%, #0e3460 100%);
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* ── Logo / header ── */
    .logo-container {
        text-align: center;
        margin-bottom: 1.5rem;
        animation: fadeIn 1s ease-in;
    }
    .main-title {
        text-align: center;
        color: #2ca3ee;
        font-size: 3rem;
        font-weight: 900;
        margin: 1.5rem 0 0.5rem 0;
        text-shadow: 0 4px 20px rgba(44,163,238,0.4);
        letter-spacing: 3px;
        animation: slideDown 0.8s ease-out;
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        letter-spacing: 0.1em;
        animation: slideDown 1s ease-out;
    }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        padding: 1.5rem 2rem;
        background: #000000;
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

    /* ── Card link wrapper ── */
    a.card-link {
        text-decoration: none !important;
        display: block;
        height: 100%;
    }
    a.card-link:hover { text-decoration: none !important; }

    /* ── Service cards ── */
    .service-card {
        background: #000000;
        border-radius: 18px;
        padding: 2.2rem 1.8rem;
        text-align: center;
        border: 2px solid rgba(44,163,238,0.25);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        animation: fadeInUp 0.6s ease-out;
    }
    .service-card:hover {
        transform: translateY(-10px);
        border-color: #2ca3ee;
        box-shadow: 0 20px 60px rgba(44,163,238,0.3);
        background: #000000;
    }

    /* SAFie card gets yellow accent on hover */
    .service-card.safire-card:hover {
        border-color: #e6fe00;
        box-shadow: 0 20px 60px rgba(230,254,0,0.2);
    }

    .service-icon {
        font-size: 4rem;
        margin-bottom: 1.2rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
    }
    .service-logo {
        width: 110px;
        height: 110px;
        object-fit: contain;
        margin: 0 auto 1.2rem auto;
        display: block;
        filter: drop-shadow(0 4px 12px rgba(44,163,238,0.4));
    }
    .service-title {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        letter-spacing: 0.02em;
    }
    /* SAFie title in brand blue */
    .service-title.safire-title { color: #2ca3ee; }

    .service-description {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        line-height: 1.65;
        margin-bottom: 1.5rem;
        min-height: 60px;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    /* SAFie badge — yellow/black */
    .badge-safire {
        background: #e6fe00;
        color: #000000 !important;
    }
    /* Admin — blue */
    .badge-admin {
        background: linear-gradient(135deg, #2ca3ee 0%, #00b8f1 100%);
    }
    /* Stats — yellow outline */
    .badge-stats {
        background: transparent;
        border: 2px solid #e6fe00;
        color: #e6fe00 !important;
    }

    /* ── Launch buttons ── */
    .launch-btn {
        display: inline-block;
        width: 100%;
        padding: 0.9rem 2rem;
        border-radius: 50px;
        font-weight: 800;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-sizing: border-box;
        text-decoration: none !important;
    }
    /* SAFie button — yellow/black */
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
    /* Default — blue gradient */
    .launch-btn-default {
        background: linear-gradient(135deg, #2ca3ee 0%, #00b8f1 100%);
        color: #ffffff !important;
        box-shadow: 0 6px 24px rgba(44,163,238,0.35);
    }
    .launch-btn-default:hover {
        background: linear-gradient(135deg, #00b8f1 0%, #2ca3ee 100%);
        box-shadow: 0 10px 36px rgba(44,163,238,0.55);
        color: #ffffff !important;
    }

    /* ── Quick-access links ── */
    .quick-link {
        color: #2ca3ee !important;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 0.95rem;
        transition: color 0.2s ease;
    }
    .quick-link:hover { color: #00b8f1 !important; text-decoration: underline !important; }

    /* ── Divider ── */
    hr { border-color: rgba(44,163,238,0.25) !important; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.85rem;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(44,163,238,0.2);
    }
    .footer strong { color: #2ca3ee; }

    /* ── Animations ── */
    @keyframes fadeIn    { from { opacity:0 } to { opacity:1 } }
    @keyframes slideDown {
        from { opacity:0; transform:translateY(-30px) }
        to   { opacity:1; transform:translateY(0) }
    }
    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(30px) }
        to   { opacity:1; transform:translateY(0) }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Service configuration
# --------------------------------------------------
SERVICES = [
    {
        "key":          "safire",
        "name":         "SAFie",
        "tagline":      "AI by SA Footballer",
        "icon":         "📰",
        "use_logo":     True,          # show assets/logo2.png instead of emoji
        "logo_path":    "assets/logo2.png",
        "description":  "AI-powered match report generation for magazines, web articles, and social media posts in seconds.",
        "url":          "https://magazine-user.onrender.com",
        "badge":        "AI · Core",
        "badge_class":  "badge-safire",
        "btn_class":    "launch-btn-safire",
        "card_class":   "service-card safire-card",
        "title_class":  "service-title safire-title",
    },
    {
        "key":          "admin",
        "name":         "Admin Dashboard",
        "tagline":      "",
        "icon":         "⚙️",
        "use_logo":     False,
        "logo_path":    "",
        "description":  "Complete analytics, user management, cost tracking, match links, and full system administration.",
        "url":          "https://magazine-admin.onrender.com",
        "badge":        "Admin",
        "badge_class":  "badge-admin",
        "btn_class":    "launch-btn-default",
        "card_class":   "service-card",
        "title_class":  "service-title",
    },
    {
        "key":          "league",
        "name":         "League & Ladder",
        "tagline":      "",
        "icon":         "🏆",
        "use_logo":     False,
        "logo_path":    "",
        "description":  "Sync and display live league ladders, standings, and team statistics directly from PlayHQ.",
        "url":          "https://ladder-hu0e.onrender.com",
        "badge":        "Stats",
        "badge_class":  "badge-stats",
        "btn_class":    "launch-btn-default",
        "card_class":   "service-card",
        "title_class":  "service-title",
    },
]

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    # ── Dual logo header ──────────────────────────────────────────
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_l:
        try:
            st.image("assets/logo2.png", width=130)   # SAFie logo
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
            st.image("assets/logo.png", width=130)    # SA Footballer logo
        except Exception:
            st.markdown("<p style='color:white;text-align:center;font-weight:700;'>SA Footballer</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats bar ────────────────────────────────────────────────
    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-number">3</div>
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

    # ── Service cards ─────────────────────────────────────────────
    cols = st.columns(3)
    for idx, svc in enumerate(SERVICES):
        with cols[idx]:
            # Build the logo/icon HTML
            logo_html = ""
            if svc["use_logo"]:
                logo_path = Path(svc["logo_path"])
                if logo_path.exists():
                    import base64
                    with open(logo_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    ext = logo_path.suffix.lstrip(".")
                    logo_html = f'<img src="data:image/{ext};base64,{b64}" class="service-logo" alt="{svc["name"]} logo" />'
                else:
                    logo_html = f'<div class="service-icon">{svc["icon"]}</div>'
            else:
                logo_html = f'<div class="service-icon">{svc["icon"]}</div>'

            # Optional tagline under title
            tagline_html = ""
            if svc["tagline"]:
                tagline_html = f'<div style="color:#e6fe00; font-size:0.72rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-top:-6px; margin-bottom:10px;">{svc["tagline"]}</div>'

            st.markdown(f"""
            <a class="card-link" href="{svc['url']}" target="_blank" rel="noopener noreferrer">
                <div class="{svc['card_class']}">
                    <div>
                        {logo_html}
                        <div class="{svc['title_class']}">{svc['name']}</div>
                        {tagline_html}
                        <div class="service-description">{svc['description']}</div>
                        <span class="badge {svc['badge_class']}">{svc['badge']}</span>
                    </div>
                    <div style="margin-top:1.5rem;">
                        <span class="launch-btn {svc['btn_class']}">🚀 Launch {svc['name']}</span>
                    </div>
                </div>
            </a>
            """, unsafe_allow_html=True)

    # ── Quick-access links ────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "<h3 style='color:#2ca3ee; border-bottom:3px solid #2ca3ee; padding-bottom:6px; display:inline-block;'>🔗 Quick Access Links</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, svc in zip([col1, col2, col3], SERVICES):
        with col:
            title_colour = "#2ca3ee" if svc["key"] == "safire" else "#ffffff"
            st.markdown(f"""
            <p style='color:{title_colour}; font-weight:800; font-size:1.05rem; margin-bottom:4px;'>
                {svc['icon']} {svc['name']}
            </p>
            <a href="{svc['url']}" target="_blank" rel="noopener noreferrer" class="quick-link">
                Open in New Tab →
            </a>
            <p style='color:rgba(255,255,255,0.45); font-size:0.85rem; margin-top:0.5rem;'>
                {svc['description'][:65]}…
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
