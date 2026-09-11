import pandas as pd
from typing import Tuple


def decide_escalation(
    customer_query: str,
    predicted_intent: str,
    similar_cases: pd.DataFrame
) -> Tuple[str, str]:
    query = customer_query.lower().strip()

    human_keywords = [
        "hacked",
        "account hacked",
        "stolen account",
        "stolen",
        "fraud",
        "fraudulent",
        "unauthorized payment",
        "unauthorised payment",
        "charged twice",
        "double charged",
        "refund",
        "money back",
        "payment dispute",
        "scam",
        "security issue",
    ]

    if any(keyword in query for keyword in human_keywords):
        return (
            "HUMAN_ESCALATION",
            "Sensitive account, payment, security, or refund issue requires human review."
        )

    if similar_cases.empty:
        return (
            "HUMAN_ESCALATION",
            "No sufficiently similar historical support case was found."
        )

    return (
        "AUTO_HANDLE",
        "A sufficiently similar historical support pattern was found."
    )
