import streamlit as st
import time
from model import FakeNewsClassifier

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return FakeNewsClassifier()

model = load_model()

# ── Header ─────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-family:DM Serif Display;'>
 <span style='color:red;'>FAKE  <span style='color:blue;'> NEWS <span style='color:black;'>DETECTOR</span>
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ── INPUT SECTION ─────────────────────────────────────────
st.subheader("📝 Enter News Content")

text_input = st.text_area(
    "",
    placeholder="Paste a news article / headline",
    height=200
)

analyse = st.button("🔍 Analyse", use_container_width=True)

st.markdown("---")

# ── RESULT SECTION ─────────────────────────────────────────
if analyse and text_input.strip():

    with st.spinner("Analyzing..."):
        time.sleep(0.3)
        result = model.predict(text_input)

    r = result['real_prob']

    # ── Verdict Logic ─────────────────────────
    if r >= 65:
        label = "✅ Likely Real"
        color = "#27ae60"
    elif r <= 35:
        label = "❌ Likely Fake"
        color = "#e74c3c"
    else:
        label = "⚠️ Uncertain"
        color = "#f39c12"

    # ── Verdict Card ─────────────────────────
    st.markdown(f"""
    <div style="padding:20px;border-radius:10px;border:2px solid {color};text-align:center;">
        <h2 style="color:{color};margin-bottom:5px;">{label}</h2>
        <p>Confidence Score: <b>{r}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress Bar ─────────────────────────
    st.progress(r / 100)

    st.markdown("---")

    # ── Metrics ─────────────────────────
    st.subheader("📊 Analysis Metrics")

    f = result['features']

    col1, col2, col3 = st.columns(3)
    col1.metric("Words", f['word_count'])
    col2.metric("Sensational", f['sensational'])
    col3.metric("CAPS %", f['caps_ratio'])

    col4, col5 = st.columns(2)
    col4.metric("Hedges", f['hedges'])
    col5.metric("Exclamations", f['exclaims'])

    st.markdown("---")

    # ── Vocabulary Signals ─────────────────────────
    st.subheader("🧠 Vocabulary Signals")

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

    st.markdown("---")

    # ── Score Breakdown ─────────────────────────
    st.subheader("📊 Score Breakdown")

    st.write(f"Real Score: {result['real_score']}")
    st.write(f"Fake Score: {result['fake_score']}")

    elif analyse:
    st.warning("⚠️ Please enter text to analyse")

    else:
    st.info("👆 Paste content above and click Analyse")

    else:
    st.info("Enter text and click Analyse")
