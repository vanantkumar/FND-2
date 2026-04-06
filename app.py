import streamlit as st
import time
from model import FakeNewsClassifier

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return FakeNewsClassifier()

model = load_model()

# ── Header ─────────────────────────────────────────
st.markdown("""
<h1 style='font-family:DM Serif Display;'>
FAKE NEWS <span style='color:red;'>DETECTOR</span>
</h1>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────
col_input, col_result = st.columns([1, 1])

# ── INPUT SECTION ─────────────────────────────────────────
with col_input:
    st.subheader("📝 Input Text")

    text_input = st.text_area(
        "Enter News",
        placeholder="Paste news article or headline...",
        height=200
    )

    analyse = st.button("Analyse")

# ── RESULT SECTION ─────────────────────────────────────────
with col_result:
    st.subheader("📊 Result")

    if analyse and text_input.strip():

        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            result = model.predict(text_input)

        r = result['real_prob']

        # ── Verdict Logic ─────────────────────────
        if r >= 65:
            label = "✅ Likely Real"
            color = "green"
        elif r <= 35:
            label = "❌ Likely Fake"
            color = "red"
        else:
            label = "⚠️ Uncertain"
            color = "orange"

        # ── Display Result ─────────────────────────
        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;border:2px solid {color};">
            <h2 style="color:{color};">{label}</h2>
            <p>Confidence: <b>{r}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ─────────────────────────
        f = result['features']

        st.write("### 📌 Analysis Metrics")
        st.write(f"Words: {f['word_count']}")
        st.write(f"Sensational Words: {f['sensational']}")
        st.write(f"CAPS Ratio: {f['caps_ratio']}%")
        st.write(f"Hedges: {f['hedges']}")
        st.write(f"Exclamations: {f['exclaims']}")

        # ── Vocabulary Signals ─────────────────────────
        st.write("### 🧠 Vocabulary Signals")

        st.write("**Fake Indicators:**")
        if result['matched_fake']:
            st.write(", ".join(result['matched_fake']))
        else:
            st.write("None")

        st.write("**Real Indicators:**")
        if result['matched_real']:
            st.write(", ".join(result['matched_real']))
        else:
            st.write("None")

        # ── Score Breakdown ─────────────────────────
        st.write("### 📊 Score Breakdown")

        st.progress(r / 100)
        st.write(f"Real Score: {result['real_score']}")
        st.write(f"Fake Score: {result['fake_score']}")

    elif analyse:
        st.warning("⚠️ Please enter text first")

    else:
        st.info("Enter text and click Analyse")
