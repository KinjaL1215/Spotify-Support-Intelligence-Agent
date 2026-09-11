import numpy as np
from .intent_classifier import assign_intent_rules, INTENT_CATEGORIES
from .llm_intent_classifier import classify_intent_llm


class HybridIntentClassifier:
    def __init__(self, vectorizer=None, svm_classifier=None, groq_client=None, confidence_threshold: float = 0.5):
        self.vectorizer = vectorizer
        self.svm_classifier = svm_classifier
        self.groq_client = groq_client
        self.confidence_threshold = confidence_threshold

    def predict(self, query: str) -> str:
        # 1. High confidence rule match
        rule_intent = assign_intent_rules(query)
        if rule_intent != "General / Other Support":
            return rule_intent

        # 2. Machine Learning Linear SVM prediction
        if self.vectorizer is not None and self.svm_classifier is not None:
            features = self.vectorizer.transform([query])
            svm_pred = self.svm_classifier.predict(features)[0]
            scores = self.svm_classifier.decision_function(features)

            if scores.ndim == 2:
                sorted_scores = np.sort(scores, axis=1)
                margin = sorted_scores[0, -1] - sorted_scores[0, -2]
                if margin >= self.confidence_threshold:
                    return svm_pred

        # 3. LLM Zero-Shot Fallback
        if self.groq_client is not None:
            return classify_intent_llm(query, self.groq_client)

        if self.vectorizer is not None and self.svm_classifier is not None:
            features = self.vectorizer.transform([query])
            return self.svm_classifier.predict(features)[0]

        return rule_intent
