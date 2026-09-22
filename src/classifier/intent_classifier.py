import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# INTENT CATEGORIES
# ============================================================

INTENT_CATEGORIES = [
    "Account / Login",
    "Premium / Billing / Subscription",
    "Playback / Streaming Issue",
    "App / Device Issue",
    "Music / Content Availability",
    "Spotify Connect / External Devices",
    "Feature Request / Product Feedback",
    "General / Other Support",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean customer text by:
    - handling missing values
    - decoding HTML entities
    - removing URLs
    - removing @mentions
    - normalizing whitespace
    """

    text = "" if pd.isna(text) else str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RULE-BASED INTENT LABELING
# ============================================================

def assign_intent_rules(text: str) -> str:

    text = str(text).lower().strip()

    # --------------------------------------------------------
    # ACCOUNT / LOGIN
    # --------------------------------------------------------

    account_patterns = [
        r"\blogin\b",
        r"\blog in\b",
        r"\blogged in\b",
        r"\bsign in\b",
        r"\bsignin\b",
        r"\bpassword\b",
        r"\busername\b",
        r"\bhacked account\b",
        r"\baccount hacked\b",
        r"\baccount access\b",
        r"\bcannot access my account\b",
        r"\bcan't access my account\b",
        r"\breset password\b",
        r"\bforgot password\b",
        r"\bemail changed\b",
        r"\bcannot login\b",
        r"\bcan't login\b",
        r"\bunable to login\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in account_patterns
    ):
        return "Account / Login"


    # --------------------------------------------------------
    # PREMIUM / BILLING / SUBSCRIPTION
    # --------------------------------------------------------

    premium_patterns = [
        r"\bpremium\b",
        r"\bsubscription\b",
        r"\bsubscribed\b",
        r"\bbilling\b",
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bpayment\b",
        r"\bpaid\b",
        r"\brefund\b",
        r"\binvoice\b",
        r"\brenew\b",
        r"\brenewal\b",
        r"\bcancel premium\b",
        r"\bfree trial\b",
        r"\bpremium plan\b",
        r"\bpremium membership\b",
        r"\bpayment failed\b",
        r"\bpayment issue\b",
        r"\bcharged twice\b",
        r"\bdouble charged\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in premium_patterns
    ):
        return "Premium / Billing / Subscription"


    # --------------------------------------------------------
    # SPOTIFY CONNECT / EXTERNAL DEVICES
    # --------------------------------------------------------

    connect_patterns = [
        r"\bspotify connect\b",
        r"\bchromecast\b",
        r"\bgoogle home\b",
        r"\bsmart speaker\b",
        r"\bsmart tv\b",
        r"\bsonos\b",
        r"\bplaystation\b",
        r"\bxbox\b",
        r"\bconnected device\b",
        r"\bexternal device\b",
        r"\bconnect to my speaker\b",
        r"\bconnect to speaker\b",
        r"\bconnect to tv\b",
        r"\bdevice connection\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in connect_patterns
    ):
        return "Spotify Connect / External Devices"


    # --------------------------------------------------------
    # APP / DEVICE ISSUE
    # --------------------------------------------------------

    device_patterns = [
        r"\bapp crash\b",
        r"\bapp crashes\b",
        r"\bcrashing\b",
        r"\bcrash\b",
        r"\biphone\b",
        r"\bipad\b",
        r"\bandroid\b",
        r"\bwindows phone\b",
        r"\bphone\b",
        r"\btablet\b",
        r"\bcomputer\b",
        r"\bdesktop\b",
        r"\binstall\b",
        r"\buninstall\b",
        r"\bupdate\b",
        r"\bdevice\b",
        r"\bcompatibility\b",
        r"\bnot opening\b",
        r"\bwon't open\b",
        r"\bcannot open\b",
        r"\bcan't open\b",
        r"\bapp not working\b",
        r"\bapplication not working\b",
        r"\bmobile app\b",
        r"\bdesktop app\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in device_patterns
    ):
        return "App / Device Issue"


    # --------------------------------------------------------
    # PLAYBACK / STREAMING ISSUE
    # --------------------------------------------------------

    playback_patterns = [
        r"\bnot playing\b",
        r"\bwon't play\b",
        r"\bwont play\b",
        r"\bcant play\b",
        r"\bcan't play\b",
        r"\bcannot play\b",
        r"\bplayback\b",
        r"\bstops playing\b",
        r"\bstopped playing\b",
        r"\bkeeps stopping\b",
        r"\bkeeps pausing\b",
        r"\bpauses\b",
        r"\bpause\b",
        r"\bskipping\b",
        r"\bskips\b",
        r"\bbuffering\b",
        r"\bbuffer\b",
        r"\bstreaming\b",
        r"\bmusic stops\b",
        r"\bsong stops\b",
        r"\bsongs stop\b",
        r"\bwrong song playing\b",
        r"\baudio problem\b",
        r"\bsound problem\b",
        r"\bmusic not playing\b",
        r"\bsongs not playing\b",
        r"\bcan't listen\b",
        r"\bcannot listen\b",
        r"\bunable to play\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in playback_patterns
    ):
        return "Playback / Streaming Issue"


    # --------------------------------------------------------
    # MUSIC / CONTENT AVAILABILITY
    # --------------------------------------------------------

    content_patterns = [
        r"\bsong unavailable\b",
        r"\bsongs unavailable\b",
        r"\btrack unavailable\b",
        r"\balbum unavailable\b",
        r"\bartist unavailable\b",
        r"\bsong missing\b",
        r"\bsongs missing\b",
        r"\balbum missing\b",
        r"\btrack missing\b",
        r"\bmissing song\b",
        r"\bmissing songs\b",
        r"\bnot available\b",
        r"\bunavailable\b",
        r"\bcan't find song\b",
        r"\bcannot find song\b",
        r"\bcant find song\b",
        r"\bcan't find track\b",
        r"\bcannot find track\b",
        r"\bwhere is this song\b",
        r"\bwhere is the song\b",
        r"\bremoved song\b",
        r"\bremoved songs\b",
        r"\bmusic missing\b",
        r"\bmissing music\b",
        r"\bnot on spotify\b",
        r"\bnot on spotify anymore\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in content_patterns
    ):
        return "Music / Content Availability"


    # --------------------------------------------------------
    # FEATURE / PRODUCT FEEDBACK
    # --------------------------------------------------------

    feature_patterns = [
        r"\bfeature request\b",
        r"\bfeature\b",
        r"\bplease add\b",
        r"\badd a feature\b",
        r"\bwould like\b",
        r"\bi wish\b",
        r"\bshould add\b",
        r"\bbring back\b",
        r"\bplease bring back\b",
        r"\bsuggestion\b",
        r"\bsuggest\b",
        r"\bfeedback\b",
        r"\bimprove\b",
        r"\bimprovement\b",
        r"\bnew feature\b",
        r"\bcan you add\b",
        r"\bcould you add\b",
        r"\bplease support\b",
        r"\badd support\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in feature_patterns
    ):
        return "Feature Request / Product Feedback"


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return "General / Other Support"


# ============================================================
# TF-IDF PIPELINE
# ============================================================

def build_tfidf_pipeline():

    return FeatureUnion([
        (
            "word",

            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=30000,
                sublinear_tf=True,
                strip_accents="unicode"
            )
        ),

        (
            "character",

            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=30000,
                sublinear_tf=True
            )
        )
    ])


# ============================================================
# TRAIN SVM CLASSIFIER
# ============================================================

def train_svm_classifier(
    classifier_data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
):

    X = classifier_data[
        "customer_text_clean"
    ].fillna("")

    y = classifier_data[
        "intent"
    ].fillna("General / Other Support")


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = build_tfidf_pipeline()

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    classifier = LinearSVC(
        C=1.5,
        class_weight="balanced"
    )

    classifier.fit(
        X_train_tfidf,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = classifier.predict(
        X_test_tfidf
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return (
        vectorizer,
        classifier,
        accuracy,
        X_test,
        y_test,
        predictions,
        X_test_tfidf
    )


# ============================================================
# SAVE TRAINED CLASSIFIER
# ============================================================

def save_classifier(
    vectorizer,
    classifier,
    model_dir
):
    """
    Save trained TF-IDF vectorizer and SVM classifier.
    """

    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    vectorizer_path = (
        model_dir /
        "tfidf_vectorizer.pkl"
    )

    classifier_path = (
        model_dir /
        "svm_classifier.pkl"
    )


    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    joblib.dump(
        classifier,
        classifier_path
    )


    print(
        "TF-IDF vectorizer saved to:",
        vectorizer_path
    )

    print(
        "SVM classifier saved to:",
        classifier_path
    )


# ============================================================
# LOAD TRAINED CLASSIFIER
# ============================================================

def load_classifier(model_dir):
    """
    Load previously trained TF-IDF vectorizer
    and SVM classifier.
    """

    model_dir = Path(model_dir)

    vectorizer_path = (
        model_dir /
        "tfidf_vectorizer.pkl"
    )

    classifier_path = (
        model_dir /
        "svm_classifier.pkl"
    )


    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {vectorizer_path}"
        )


    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier not found: {classifier_path}"
        )


    vectorizer = joblib.load(
        vectorizer_path
    )

    classifier = joblib.load(
        classifier_path
    )


    print(
        "TF-IDF vectorizer loaded."
    )

    print(
        "SVM classifier loaded."
    )


    return vectorizer, classifier