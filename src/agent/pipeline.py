from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.retrieval.dense_retriever import retrieve_similar_cases
from src.escalation.decision import decide_escalation
from src.agent.support_agent import generate_support_response


@dataclass
class AgentResult:
    query: str
    predicted_intent: str
    escalation: str
    escalation_reason: str
    response_text: str
    similar_cases: pd.DataFrame


class SupportAgentPipeline:
    def __init__(
        self,
        rag_data: pd.DataFrame,
        embedding_model: SentenceTransformer,
        index: Any,
        vectorizer: Any,
        classifier: Any,
        client: Optional[Groq] = None,
        top_k: int = 5
    ):
        self.rag_data = rag_data
        self.embedding_model = embedding_model
        self.index = index
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.client = client
        self.top_k = top_k

    def run(self, query: str) -> AgentResult:
        query_clean = query.strip()
        
        # 1. Intent Classification
        features = self.vectorizer.transform([query_clean])
        predicted_intent = self.classifier.predict(features)[0]

        # 2. Dense Semantic Retrieval
        similar_cases = retrieve_similar_cases(
            query_clean,
            self.embedding_model,
            self.index,
            self.rag_data,
            k=self.top_k
        )

        # 3. Escalation Decision
        escalation, escalation_reason = decide_escalation(
            query_clean,
            predicted_intent,
            similar_cases
        )

        # 4. Grounded Support Response
        response_text = generate_support_response(
            query_clean,
            self.embedding_model,
            self.index,
            self.rag_data,
            self.client,
            k=self.top_k
        )

        return AgentResult(
            query=query_clean,
            predicted_intent=predicted_intent,
            escalation=escalation,
            escalation_reason=escalation_reason,
            response_text=response_text,
            similar_cases=similar_cases
        )
