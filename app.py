"""
app.py
Streamlit demo: paste text, get a 3-way prediction (Human / Gemini / GPT-OSS)
with confidence per class, plus a watermark check for English text.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="LLM Attribution Detector", layout="centered")
st.title("Human vs. Gemini vs. GPT-OSS: Text Attribution")
st.caption("Stylometric + embedding-based classifier (EN/DE) — attributing which LLM (if any) generated a text")

text = st.text_area("Paste a news snippet (EN/DE):", height=150)
language = st.selectbox("Language", ["en", "de"])

# TODO: load your trained model here, e.g.:
# clf, scaler = joblib.load("outputs/attribution_clf.pkl")
# embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

if st.button("Analyze") and text.strip():
    st.subheader("Attribution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Human", "(wire up model)")
    with col2:
        st.metric("Gemini", "(wire up model)")
    with col3:
        st.metric("GPT-OSS (Groq)", "(wire up model)")

    if language == "en":
        st.subheader("Watermark check (toy demo)")
        st.caption(
            "Statistical green-list/red-list detection — same family of technique "
            "as SynthID-Text and Anthropic's production Claude watermark. "
            "Short texts won't carry enough signal to detect reliably."
        )
        st.write("*(wire up watermark_demo.detect_watermark here)*")

st.markdown("---")
st.caption(
    "Built as a portfolio project. Human text: ag_news (EN), gnad10 (DE). "
    "AI text: same-topic rewrites from Gemini and GPT-OSS 120B (via Groq). "
    "No raw dataset files redistributed."
)

st.markdown("---")
st.caption(
    "Built as a portfolio project. Human text: ag_news (EN), gnad10 (DE), CLUE tnews (ZH). "
    "AI text: LLM-generated same-topic rewrites. No raw dataset files redistributed."
)
