import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Netflix Content Intelligence Platform",
    page_icon="🎬",
    layout="wide"
)

# =========================
# NETFLIX STYLE BACKGROUND (FIXED)
# =========================
st.markdown("""
<style>

/* Background image */
.stApp {
    background: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* DARK OVERLAY FIX (IMPORTANT) */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.78);
    z-index: 0;
    pointer-events: none;
}

/* KEEP CONTENT ABOVE BACKGROUND */
.block-container {
    position: relative;
    z-index: 1;
}

/* TEXT */
h1, h2, h3, p, span {
    color: white !important;
}

/* METRIC CARDS (NETFLIX STYLE) */
div[data-testid="stMetric"] {
    background: rgba(24, 24, 24, 0.85);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.5);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION (FIXED - NO RAW HTML BUG)
# =========================
st.markdown("""
# 🎬 Netflix Content Intelligence Platform

### AI Recommendation Engine • Semantic Search • Clustering • Analytics Dashboard

---

""")

st.markdown("""
<div style="
    background: linear-gradient(120deg, #E50914, #000000);
    padding: 40px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
">

<h2 style="color:white;">🚀 Welcome to Netflix Intelligence System</h2>

<p style="font-size:16px; opacity:0.9;">
This platform replicates Netflix-level AI recommendation intelligence using ML models.
</p>

<div style="
    display:inline-block;
    background:#E50914;
    padding:10px 18px;
    border-radius:8px;
    font-weight:bold;
    margin-top:10px;
">
▶ Explore AI System
</div>

</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# DESCRIPTION
# =========================
st.markdown("""
### 📌 What this platform includes

- 🎯 Recommendation Engine (TF-IDF + BERT)
- 🔍 Semantic Search
- 🧠 Content Clustering
- 📊 Model Comparison
- 📈 A/B Testing & Insights
""")

st.divider()

# =========================
# METRICS (MATCH PIC 1 STYLE)
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎬 Total Titles", "8,800+")

with col2:
    st.metric("🧠 AI Models", "2")

with col3:
    st.metric("⚙️ ML Algorithms", "4")

with col4:
    st.metric("📊 Pages", "8+")

st.divider()

# =========================
# FEATURE CARDS
# =========================
st.markdown("## ✨ Features")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
### 🎯 Recommendation Engine
- TF-IDF similarity
- BERT semantic search
- Ranking system

### 🔍 Semantic Search
- Natural language input
- Smart retrieval system
""")

with c2:
    st.markdown("""
### 🧠 Clustering
- KMeans clustering
- PCA visualization
- Pattern discovery

### 📊 Analytics
- Model comparison
- A/B testing
- Performance insights
""")

st.divider()

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; padding:20px; opacity:0.8;">
🚀 Developed by <b>Divyansh Agarwal</b>
</div>
""", unsafe_allow_html=True)