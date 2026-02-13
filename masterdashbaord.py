import streamlit as st
import os
from pathlib import Path

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="South Australian Footballer - Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Main title */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 2rem 0 1rem 0;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
        animation: slideDown 0.8s ease-out;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 3rem;
        animation: slideDown 1s ease-out;
    }
    
    /* Service card */
    .service-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .service-card:hover {
        transform: translateY(-10px);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.4);
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.1));
    }
    
    /* Service icon */
    .service-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }
    
    /* Service title */
    .service-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Service description */
    .service-description {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        min-height: 60px;
    }
    
    /* Launch button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Stats bar */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 3rem 0;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3b82f6;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .stat-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Service URLs Configuration
# --------------------------------------------------
# Update these URLs with your actual Render deployment URLs
SERVICES = {
    "magazine": {
        "name": "Magazine Automation",
        "icon": "📰",
        "description": "AI-powered match report generation for magazines, web articles, and social media",
        "url": "https://magazine-user.onrender.com",  # Update with your URL
        "badge": "Core"
    },
    "admin": {
        "name": "Admin Dashboard",
        "icon": "⚙️",
        "description": "Complete analytics, user management, cost tracking, and system administration",
        "url": "https://magazine-admin.onrender.com",  # Update with your URL
        "badge": "Admin"
    },
    "league": {
        "name": "League & Ladder",
        "icon": "🏆",
        "description": "Sync and display league ladders, standings, and team statistics from PlayHQ",
        "url": "https://ladder-hu0e.onrender.com",  # Update with your URL
        "badge": "Stats"
    }
}

# --------------------------------------------------
# Main Dashboard
# --------------------------------------------------
def main():
    # Logo
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image(str(logo_path), width=250)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="logo-container">🏈</div>', unsafe_allow_html=True)
    
    # Main Title
    st.markdown('<h1 class="main-title">SOUTH AUSTRALIAN FOOTBALLER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Master Dashboard · Content Management Suite</p>', unsafe_allow_html=True)
    
    # Stats Bar
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
    
    # Service Cards
    cols = st.columns(3)
    
    for idx, (key, service) in enumerate(SERVICES.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="service-card">
                <div>
                    <div class="service-icon">{service['icon']}</div>
                    <div class="service-title">{service['name']}</div>
                    <div class="service-description">{service['description']}</div>
                    <span class="badge">{service['badge']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(f"🚀 Launch {service['name']}", key=f"btn_{key}", use_container_width=True):
                st.markdown(f'<meta http-equiv="refresh" content="0;url={service["url"]}">', unsafe_allow_html=True)
                st.info(f"🔗 Redirecting to {service['name']}...")
                st.markdown(f"**Direct Link:** [{service['url']}]({service['url']})")
    
    # Quick Access Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🔗 Quick Access Links")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📰 Magazine Automation")
        st.markdown(f"[Open Service →]({SERVICES['magazine']['url']})")
        st.caption("Generate professional match reports")
    
    with col2:
        st.markdown("#### ⚙️ Admin Dashboard")
        st.markdown(f"[Open Service →]({SERVICES['admin']['url']})")
        st.caption("Manage users and view analytics")
    
    with col3:
        st.markdown("#### 🏆 League & Ladder")
        st.markdown(f"[Open Service →]({SERVICES['league']['url']})")
        st.caption("Track team standings")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🏈 <strong>South Australian Footballer</strong> · Professional Sports Content Management</p>
        <p style="margin-top: 0.5rem; font-size: 0.85rem;">
            Powered by AI · Built with Streamlit · Deployed on Render
        </p>
        <p style="margin-top: 1rem; font-size: 0.8rem; color: #475569;">
            © 2025 South Australian Footballer. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()
