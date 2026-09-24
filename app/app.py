import sys
from pathlib import Path

# Add project root to sys.path so src.* modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.agent.llm_client import create_groq_client, GROQ_API_KEY

from src.retrieval.dense_retriever import (
    load_embedding_model,
    build_or_load_rag_index,
    DEFAULT_EMBEDDING_MODEL,
)

# IMPORTANT:
# Load saved classifier instead of training it every time
from src.classifier.intent_classifier import load_classifier

from src.agent.pipeline import SupportAgentPipeline


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAG_FILE = DATA_DIR / "spotify_conversations.csv"

CLASSIFIER_FILE = DATA_DIR / "spotify_classifier_balanced.csv"

EMBEDDING_FILE = DATA_DIR / "embeddings.npy"

FAISS_FILE = DATA_DIR / "spotify_faiss.index"

# Saved classifier files
MODEL_DIR = DATA_DIR / "models"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hiver | Spotify Support AI Agent",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SVG ICONS
# ============================================================

SPOTIFY_LOGO_SVG = """
<svg width="26" height="26" viewBox="0 0 24 24"
fill="#1DB954" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2C6.477 2 2 6.477 2 12C2 17.523 6.477 22 12 22C17.523 22 22 17.523 22 12C22 6.477 17.523 2 12 2ZM16.586 16.424C16.398 16.732 15.998 16.828 15.69 16.64C13.256 15.152 10.18 14.814 6.758 15.596C6.41 15.676 6.064 15.454 5.984 15.106C5.904 14.758 6.126 14.412 6.474 14.332C10.238 13.472 13.652 13.856 16.37 15.528C16.678 15.716 16.774 16.116 16.586 16.424ZM17.818 13.634C17.584 14.014 17.086 14.134 16.706 13.9C13.978 12.224 9.878 11.744 6.646 12.726C6.22 12.856 5.768 12.616 5.638 12.19C5.508 11.764 5.748 11.312 6.174 11.182C9.874 10.058 14.414 10.592 17.552 12.522C17.932 12.754 18.052 13.254 17.818 13.634ZM17.95 10.748C14.654 8.79 9.214 8.61 6.066 9.566C5.562 9.718 5.03 9.432 4.878 8.928C4.726 8.424 5.012 7.892 5.516 7.74C9.13 6.642 15.132 6.848 18.916 9.094C19.368 9.362 19.516 9.944 19.248 10.396C18.98 10.846 18.4 10.996 17.95 10.748Z"/>
</svg>
"""

BOT_AVATAR_SVG = """
<svg width="22" height="22" viewBox="0 0 24 24"
fill="none" stroke="#1ed760" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M12 8V4H8"/>
<rect width="16" height="12" x="4" y="8" rx="2"/>
<path d="M2 14h2"/>
<path d="M20 14h2"/>
<path d="M15 13v2"/>
<path d="M9 13v2"/>
</svg>
"""

USER_AVATAR_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24"
fill="none" stroke="#58a6ff" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>
<circle cx="12" cy="7" r="4"/>
</svg>
"""


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans',
        -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }

    .block-container {
        max-width: 820px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e1b4b 50%,
            #064e3b 100%
        );

        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(29, 185, 84, 0.15);
        color: #1ed760;
        border: 1px solid rgba(29, 185, 84, 0.35);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.35rem;
        margin-bottom: 0;
        line-height: 1.45;
    }

    .reply-card {
        background: #161b22;
        border-left: 4px solid #1DB954;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
        border-radius: 14px;
        padding: 1.4rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .reply-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #21262d;
        color: #1ed760;
        font-weight: 700;
        font-size: 0.95rem;
    }

    .reply-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .reply-content {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #f0f6fc;
        white-space: pre-wrap;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .pill-green {
        background: rgba(29, 185, 84, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.35);
    }

    .pill-amber {
        background: rgba(210, 153, 34, 0.15);
        color: #e3b341;
        border: 1px solid rgba(210, 153, 34, 0.4);
    }

    .pill-blue {
        background: rgba(88, 166, 255, 0.15);
        color: #79c0ff;
        border: 1px solid rgba(88, 166, 255, 0.35);
    }

    .pill-red {
        background: rgba(248, 81, 73, 0.15);
        color: #ff7b72;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }

    .case-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    .case-header {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.86rem;
        color: #58a6ff;
        margin-bottom: 5px;
    }

    .stTextArea textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        color: #f0f6fc !important;
        font-size: 0.98rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #1ed760 !important;
        box-shadow: 0 0 0 1px #1ed760 !important;
    }

    div[data-testid="stFormSubmitButton"] > button,
    div.stFormSubmitButton > button,
    div.stButton > button:first-child {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        min-height: 52px !important;
        padding: 0.85rem 2rem !important;
        background-color: #1DB954 !important;
        color: #000000 !important;
        border: none !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 16px rgba(29, 185, 84, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stFormSubmitButton > button:hover,
    div.stButton > button:first-child:hover {
        background-color: #1ed760 !important;
        color: #000000 !important;
        transform: scale(1.015) !important;
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.5) !important;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #1ed760;
        display: inline-block;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SUPPORT AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_support_agent():

    # Load RAG data
    rag_data = pd.read_csv(RAG_FILE)

    # Classifier dataset is no longer needed for training here.
    # It was already used during offline training.
    
    # Load embedding model
    embedding_model = load_embedding_model(
        DEFAULT_EMBEDDING_MODEL
    )

    # Load existing FAISS index
    _, index = build_or_load_rag_index(
        rag_data,
        embedding_model,
        EMBEDDING_FILE,
        FAISS_FILE
    )

    # Create Groq client
    client = create_groq_client()

    # ========================================================
    # LOAD SAVED CLASSIFIER
    # ========================================================

    vectorizer, classifier = load_classifier(
        MODEL_DIR
    )

    # Accuracy was measured during offline training.
    # We don't retrain or re-evaluate during Streamlit startup.
    accuracy = None

    # Create pipeline
    pipeline = SupportAgentPipeline(
        rag_data=rag_data,
        embedding_model=embedding_model,
        index=index,
        vectorizer=vectorizer,
        classifier=classifier,
        client=client,
        top_k=5
    )

    return {
        "pipeline": pipeline,
        "rag_data": rag_data,
        "accuracy": accuracy,
        "embedding_model_name": DEFAULT_EMBEDDING_MODEL
    }


# ============================================================
# STARTUP
# ============================================================

try:

    with st.spinner("Preparing Spotify Support Agent..."):

        agent_bundle = load_support_agent()

        pipeline = agent_bundle["pipeline"]

except Exception as err:

    st.error("Could not load support agent pipeline.")

    st.exception(err)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("🎧 Hiver Agent")

    st.caption(
        "AI Customer Support Assistant for Spotify"
    )

    if GROQ_API_KEY:

        st.markdown(
            '<span class="pill pill-green">'
            '🟢 Groq LLM Active'
            '</span>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<span class="pill pill-amber">'
            '🟡 Retrieval Mode'
            '</span>',
            unsafe_allow_html=True
        )

    st.divider()

    st.caption(
        f"**Knowledge Cases:** "
        f"`{len(agent_bundle['rag_data']):,}`"
    )

    # Accuracy is not recalculated during app startup
    if agent_bundle["accuracy"] is not None:

        st.caption(
            f"**Model Accuracy:** "
            f"`{agent_bundle['accuracy']:.1%}`"
        )

    else:

        st.caption(
            "**Classifier:** `Pre-trained SVM`"
        )

    st.caption(
        f"**Embedding:** "
        f"`{agent_bundle['embedding_model_name']}`"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""<div class="hero-card">
<div>
<div class="hero-badge">
<span class="pulse-dot"></span>
Spotify Support Intelligence
</div>
<h1 class="hero-title">
{SPOTIFY_LOGO_SVG}
Customer Support Assistant
</h1>
<p class="hero-subtitle">
Ask any Spotify support question to receive an instant, grounded response powered by historical support intelligence.
</p>
</div>
</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_query" not in st.session_state:

    st.session_state["current_query"] = ""


# ============================================================
# EXAMPLES
# ============================================================

examples = [

    (
        "💳 Double Charge",
        "I was charged twice on my credit card for Spotify Premium."
    ),

    (
        "🔐 Login Issue",
        "I cannot log in to my Spotify account after resetting my password."
    ),

    (
        "🎵 Songs Buffering",
        "Songs keep pausing and buffering every few seconds on mobile."
    ),

    (
        "🔊 Connect Speaker",
        "Cannot connect Spotify to my smart speaker over Wi-Fi."
    ),

]


st.markdown(
    "<p style='font-size: 0.85rem; color: #8b949e; "
    "margin-bottom: 6px;'>"
    "💡 <strong>Quick Test Inquiries:</strong>"
    "</p>",
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


for i, (label, prompt_text) in enumerate(examples):

    with [c1, c2, c3, c4][i]:

        if st.button(
            label,
            key=f"ex_{i}",
            use_container_width=True
        ):

            st.session_state["current_query"] = prompt_text


# ============================================================
# SUPPORT FORM
# ============================================================

with st.form("support_form"):

    user_message = st.text_area(
        "Customer message",
        value=st.session_state["current_query"],
        height=130,
        placeholder=(
            "Enter a Spotify customer question "
            "(e.g. 'I paid for Premium but my "
            "account still shows Free')..."
        ),
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button(
        "⚡ Generate Support Reply",
        use_container_width=True
    )


# ============================================================
# PROCESS QUERY
# ============================================================

if submitted:

    query = user_message.strip()

    if not query:

        st.warning(
            "Please enter a customer message."
        )

    else:

        with st.spinner(
            "Finding best grounded resolution..."
        ):

            result = pipeline.run(query)


        # ====================================================
        # RESPONSE
        # ====================================================

        st.markdown(
            f"""<div class="reply-card">
<div class="reply-header">
<div class="reply-title-row">
{BOT_AVATAR_SVG}
<span>Spotify Support Reply</span>
</div>
<span style="font-size: 0.78rem; color: #8b949e;">Verified Grounding</span>
</div>
<div class="reply-content">{result.response_text}</div>
</div>""",
            unsafe_allow_html=True
        )


        # ====================================================
        # INTENT + ESCALATION
        # ====================================================

        meta_col1, meta_col2 = st.columns(2)

        with meta_col1:
            st.markdown(
                f"**Intent Category:** <span class='pill pill-blue'>📋 {result.predicted_intent}</span>",
                unsafe_allow_html=True
            )

        with meta_col2:
            if result.escalation == "HUMAN_ESCALATION":
                st.markdown(
                    f"**Escalation Policy:** <span class='pill pill-red'>⚠️ Human Review ({result.escalation_reason})</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "**Escalation Policy:** <span class='pill pill-green'>✅ Auto-Handled</span>",
                    unsafe_allow_html=True
                )


        # ====================================================
        # SUPPORTING CASES
        # ====================================================

        with st.expander(
            "📚 View Supporting Historical Spotify Cases"
        ):
            for idx, row in result.similar_cases.head(3).iterrows():
                st.markdown(
                    f"""<div class="case-card">
<div class="case-header">
{USER_AVATAR_SVG}
<strong>Customer:</strong> {row['customer_query']}
</div>
<div style="font-size: 0.85rem; color: #8b949e; padding-left: 24px; margin-top: 6px;">
<strong>Spotify Agent:</strong> {row['support_response']}
</div>
</div>""",
                    unsafe_allow_html=True
                )