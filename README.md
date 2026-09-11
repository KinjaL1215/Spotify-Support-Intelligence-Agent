# 🎧 Spotify Support Intelligence Agent

An end-to-end, grounded customer support assistant designed for Spotify support workflows. The system integrates **multi-class intent classification**, **dense semantic vector retrieval (FAISS + Sentence Transformers)**, **safety escalation policy guardrails**, and **grounded LLM response generation (Groq)**.

---

## 🌟 Key Features

- 🎯 **Intent Classification Engine**: Classifies incoming customer tickets into 8 distinct Spotify support categories using **Linear Support Vector Classifier (LinearSVC)** with word and character-level TF-IDF feature unions (85.5% test accuracy).
- 🧠 **Grounded Semantic Retrieval (RAG)**: Indexes 42,700+ historical Spotify customer support interactions from the Twitter Customer Support (TWCS) dataset using dense 384-dimensional embeddings (`all-MiniLM-L6-v2`) and **FAISS IndexFlatL2**.
- 🛡️ **Safety Escalation Policy**: Automated detection for sensitive inquiries (compromised accounts, unauthorized charges, double billing disputes, and low similarity cases) requiring human agent review.
- ⚡ **Grounded Response Generation**: Drafts concise, empathetic, customer-ready replies via **Groq** using verified historical support interactions as ground-truth context.
- 🎨 **Modern Streamlit Web UI**: Sleek, responsive interface with instant quick-test prompts, live classification badges, policy indicators, and collapsible grounding evidence.

---

## 🏗️ System Architecture

```
                      +-----------------------------+
                      |   Inbound Customer Ticket   |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
    +-------------------------+             +-------------------------+
    |  Intent Classification  |             |  FAISS Semantic Search  |
    | (TF-IDF + Linear SVM)   |             | (all-MiniLM-L6-v2 RAG)  |
    +------------+------------+             +------------+------------+
                 |                                       |
                 |       +-----------------------+       |
                 +------>|  Escalation Policy    |<------+
                         |  (Guardrail Engine)   |
                         +-----------+-----------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
       [HUMAN_ESCALATION]                          [AUTO_HANDLE]
  Flag for human agent review              Ground historical evidence cases
                                                         |
                                                         v
                                            +-------------------------+
                                            |  Groq LLM Generation    |
                                            | (Grounded Support Reply)|
                                            +-------------------------+
```

---

## 📁 Project Structure

```
Spotify-Support-Intelligence-Agent/
├── app/
│   └── app.py                                 # Streamlit UI & interactive chat logic
├── data/
│   ├── spotify_conversations.csv              # Cleaned Spotify customer-response pairs (RAG)
│   ├── spotify_classifier_balanced.csv        # Stratified intent classification dataset
│   ├── embeddings.npy                         # Precomputed 384-d sentence embeddings
│   ├── spotify_faiss.index                    # FAISS vector similarity index
│   ├── classifier_predictions.csv             # Model evaluation predictions
│   └── classifier_wrong_predictions.csv       # Misclassified sample error logs
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── llm_client.py                      # Groq LLM API client wrapper
│   │   ├── pipeline.py                        # End-to-end support agent pipeline
│   │   └── support_agent.py                   # Grounded response synthesis logic
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py               # TF-IDF + LinearSVC training & inference
│   │   ├── hybrid_intent_classifier.py        # Hybrid classification logic
│   │   └── llm_intent_classifier.py           # LLM-based fallback classifier
│   ├── escalation/
│   │   ├── __init__.py
│   │   └── decision.py                        # Safety escalation & guardrails engine
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_intent.py                 # Classification evaluation metrics
│   └── retrieval/
│       ├── __init__.py
│       └── dense_retriever.py                 # SentenceTransformer & FAISS retriever
├── tests/
│   ├── __init__.py
│   └── test_agent.py                          # Unit and pipeline tests
├── main.py                                    # Interactive CLI mode entry point
├── streamlit_app.py                           # Root entry point for Streamlit deployment
├── requirements.txt                           # Python project dependencies
├── .env                                       # Local API secrets (GROQ_API_KEY)
├── .gitignore                                 # Git ignore patterns
└── README.md                                  # Project documentation
```


---

## 🏷️ Supported Intent Categories

1. 🔐 **Account / Login**: Password resets, locked accounts, email changes, credentials.
2. 💳 **Premium / Billing / Subscription**: Double billing, refund requests, plan upgrades/cancellations.
3. 🎵 **Playback / Streaming Issue**: Buffering, songs pausing, playback errors.
4. 📱 **App / Device Issue**: Mobile/desktop application crashes, installation, updates.
5. 🔊 **Spotify Connect / External Devices**: Smart speakers (Sonos, Google Home, Alexa), TVs, consoles.
6. 🚫 **Music / Content Availability**: Missing tracks, unavailable albums, country restrictions.
7. 💡 **Feature Request / Product Feedback**: Suggestions, missing UI features, feedback.
8. 📋 **General / Other Support**: General inquiries and greetings.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- (Optional) [Groq API Key](https://console.groq.com) for response generation.

### 2. Environment Setup

Clone or navigate to the project directory and create a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

*(Note: Groq is optional. Without an API key, the system runs in offline retrieval mode, displaying verified historical resolutions).*

---

## 💻 Running the Application

### Option A: Streamlit Web UI (Recommended)

Launch the web interface:

```powershell
.\venv\Scripts\activate
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

### Option B: Interactive CLI Mode

Run the terminal-based interactive agent loop:

```powershell
.\venv\Scripts\activate
python main.py
```

Enter customer queries at the prompt. Type `exit` to quit.

---

## 📊 Classification Model Performance

The classifier uses a **Linear Support Vector Machine (`LinearSVC`)** trained on word and character-level TF-IDF n-grams:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **85.5%** |
| **Macro Average F1** | **0.84** |
| **Weighted Average F1** | **0.85** |
| **Top Intent Precision (Billing)** | **96.0%** |
| **Device Connection Precision** | **100.0%** |

---

## 🛡️ Escalation Rules

Tickets are automatically flagged for **Human Escalation** if:
- Sensitive keywords are detected: `hacked`, `unauthorized payment`, `charged twice`, `refund`, `fraud`, `scam`, `security issue`.
- Semantic similarity distance exceeds safe confidence thresholds (zero evidence match).
