import os
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_FILE = DATA_DIR / "twcs.csv"

RAG_FILE = DATA_DIR / "spotify_conversations.csv"
CLASSIFIER_FILE = DATA_DIR / "spotify_classifier_balanced.csv"
EMBEDDING_FILE = DATA_DIR / "embeddings.npy"
FAISS_FILE = DATA_DIR / "spotify_faiss.index"
PREDICTION_FILE = DATA_DIR / "classifier_predictions.csv"
WRONG_FILE = DATA_DIR / "classifier_wrong_predictions.csv"

from src.classifier.intent_classifier import (
    clean_text,
    assign_intent_rules,
    train_svm_classifier,
    INTENT_CATEGORIES,
)
from src.retrieval.dense_retriever import (
    load_embedding_model,
    build_or_load_rag_index,
    DEFAULT_EMBEDDING_MODEL,
)
from src.agent.llm_client import create_groq_client, GROQ_API_KEY
from src.agent.pipeline import SupportAgentPipeline
from src.evaluation.evaluate_intent import evaluate_classifier


def load_dataset():
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded. Shape:", df.shape)
    return df


def build_spotify_conversations(df):
    print("\n" + "=" * 70)
    print("BUILDING SPOTIFY SUPPORT CONVERSATIONS")
    print("=" * 70)

    spotify_df = df[df["author_id"] == "SpotifyCares"].copy()
    spotify_replies = spotify_df[spotify_df["in_response_to_tweet_id"].notna()].copy()

    conversation_df = spotify_replies.merge(
        df[["tweet_id", "text", "author_id", "inbound"]],
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_spotify", "_customer")
    ).drop_duplicates()

    conversation_df["customer_text_clean"] = conversation_df["text_customer"].apply(clean_text)
    conversation_df["spotify_text_clean"] = conversation_df["text_spotify"].apply(clean_text)

    conversation_df = conversation_df[
        (conversation_df["customer_text_clean"] != "") &
        (conversation_df["spotify_text_clean"] != "")
    ].copy()

    print("Final usable conversations:", len(conversation_df))
    return conversation_df


def create_rag_data(conversation_df):
    rag_data = conversation_df[["customer_text_clean", "spotify_text_clean"]].copy()
    rag_data.rename(
        columns={"customer_text_clean": "customer_query", "spotify_text_clean": "support_response"},
        inplace=True
    )
    rag_data.to_csv(RAG_FILE, index=False)
    print("\nRAG records:", len(rag_data))
    return rag_data


def create_classifier_dataset(conversation_df, random_state=42):
    print("\n" + "=" * 70)
    print("CREATING CLASSIFIER DATASET")
    print("=" * 70)

    pool_size = min(10000, len(conversation_df))
    candidate_pool = conversation_df[["customer_text_clean", "spotify_text_clean"]].sample(
        n=pool_size, random_state=random_state
    ).reset_index(drop=True)

    candidate_pool["intent"] = candidate_pool["customer_text_clean"].apply(assign_intent_rules)

    target_per_intent = {
        "General / Other Support": 400,
        "Premium / Billing / Subscription": 150,
        "App / Device Issue": 150,
        "Account / Login": 100,
        "Feature Request / Product Feedback": 80,
        "Playback / Streaming Issue": 50,
        "Spotify Connect / External Devices": 40,
        "Music / Content Availability": 30,
    }

    selected_parts = []
    for intent, target in target_per_intent.items():
        rows = candidate_pool[candidate_pool["intent"] == intent]
        available = len(rows)
        selected_count = min(target, available)
        if selected_count > 0:
            selected = rows.sample(n=selected_count, random_state=random_state)
            selected_parts.append(selected)

    classifier_data = pd.concat(selected_parts, ignore_index=True)
    classifier_data = classifier_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    classifier_data = classifier_data[["customer_text_clean", "spotify_text_clean", "intent"]]
    classifier_data.to_csv(CLASSIFIER_FILE, index=False)

    print("Final classifier dataset:", len(classifier_data))
    return classifier_data


def run_interactive_test(pipeline: SupportAgentPipeline):
    print("\n" + "=" * 70)
    print("INTERACTIVE SUPPORT AGENT")
    print("=" * 70)
    print("Enter one customer message at a time. Type 'exit' to close.")

    while True:
        customer_query = input("\nCustomer message: ").strip()
        if not customer_query:
            continue
        if customer_query.lower() in {"exit", "quit", "q"}:
            print("\nSupport agent closed.")
            break

        result = pipeline.run(customer_query)

        print("\nPredicted intent:", result.predicted_intent)
        print("\nDecision:", result.escalation)
        print("Reason:", result.escalation_reason)
        print("\nReply:\n", result.response_text)


def main():
    if RAG_FILE.exists() and CLASSIFIER_FILE.exists():
        print("Loading pre-processed data...")
        rag_data = pd.read_csv(RAG_FILE)
        classifier_data = pd.read_csv(CLASSIFIER_FILE)
    else:
        df = load_dataset()
        conversation_df = build_spotify_conversations(df)
        rag_data = create_rag_data(conversation_df)
        classifier_data = create_classifier_dataset(conversation_df)

    print("\nLoading embedding model:", DEFAULT_EMBEDDING_MODEL)
    embedding_model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    embedding_matrix, index = build_or_load_rag_index(rag_data, embedding_model, EMBEDDING_FILE, FAISS_FILE)

    client = create_groq_client()

    print("\nTraining intent classifier...")
    (
        vectorizer,
        classifier,
        accuracy,
        X_test,
        y_test,
        predictions,
        X_test_tfidf
    ) = train_svm_classifier(classifier_data)

    eval_results = evaluate_classifier(
        classifier=classifier,
        X_test=X_test,
        y_test=y_test,
        X_test_tfidf=X_test_tfidf,
        predictions=predictions,
        prediction_file=PREDICTION_FILE,
        wrong_file=WRONG_FILE
    )

    print(f"\nModel Accuracy: {eval_results['accuracy']:.2%}")
    print("\nClassification Report:\n", eval_results["report"])

    pipeline = SupportAgentPipeline(
        rag_data=rag_data,
        embedding_model=embedding_model,
        index=index,
        vectorizer=vectorizer,
        classifier=classifier,
        client=client,
        top_k=5
    )

    print("\nPipeline Ready! You can also run the web app with:")
    print("  streamlit run app/app.py")


if __name__ == "__main__":
    main()
