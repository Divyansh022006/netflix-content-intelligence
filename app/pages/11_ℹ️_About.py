import streamlit as st

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="About - Netflix AI System",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("ℹ️ About Netflix Content Intelligence System")
st.markdown("### End-to-End AI + ML Recommendation Platform")

st.markdown("---")

# =========================
# PROJECT OVERVIEW
# =========================

st.subheader("🧠 Project Overview")

st.write("""
This project is a full-scale AI-powered Netflix content intelligence system.
It combines Machine Learning, NLP, and Deep Learning techniques to provide:

- Personalized recommendations
- Semantic search (Google-like Netflix search)
- Content clustering
- Model explainability
- A/B testing simulation
""")

# =========================
# ARCHITECTURE
# =========================

st.subheader("⚙️ System Architecture")

st.markdown("""
1. Netflix Dataset (Movies & TV Shows)  
2. Data Cleaning & Feature Engineering  
3. TF-IDF Vectorization (Keyword-based model)  
4. BERT Embeddings (Semantic understanding model)  
5. Similarity Matrix Computation  
6. Recommendation Engine  
7. Streamlit Web App (Frontend UI)  
""")

# =========================
# MODELS
# =========================

st.subheader("🤖 AI Models Used")

st.markdown("""
- **TF-IDF Model** → Keyword-based similarity  
- **BERT Model** → Semantic context understanding  
""")

# =========================
# FEATURES
# =========================

st.subheader("📊 Features")

st.markdown("""
- 🎬 AI Recommendation System  
- 🔍 Semantic Search Engine  
- 🧠 Cluster Explorer  
- 🎯 Explainability Engine  
- 🧪 A/B Testing Simulation  
- 🌍 Content Explorer  
- 🔥 Trending Simulation  
""")

# =========================
# TECH STACK
# =========================

st.subheader("🚀 Tech Stack")

st.markdown("""
- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- Sentence Transformers / BERT  
- Streamlit  
- Matplotlib  
""")

# =========================
# HIGHLIGHTS
# =========================

st.subheader("⭐ Project Highlights")

st.success("""
✔ End-to-End ML System  
✔ Dual Recommendation Engines (TF-IDF + BERT)  
✔ Real-Time Semantic Search  
✔ Explainable AI Layer  
✔ A/B Testing Simulation  
✔ Production-style Streamlit App  
""")

# =========================
# AUTHOR
# =========================

st.markdown("---")

st.subheader("👨‍💻 Developed By")

st.success("Divyansh Agarwal")

st.markdown("---")

st.markdown("🚀 Built as a Full-Stack AI/ML Portfolio Project")