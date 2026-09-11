from src.agent.llm_client import GROQ_MODEL, create_groq_client
from .intent_classifier import INTENT_CATEGORIES


def classify_intent_llm(query: str, client=None) -> str:
    if client is None:
        client = create_groq_client()
    if client is None:
        return "General / Other Support"

    categories_list = "\n".join([f"- {cat}" for cat in INTENT_CATEGORIES])
    prompt = f"""You are a customer support intent classifier for Spotify.
Classify the following customer query into EXACTLY ONE of these categories:
{categories_list}

Rules:
- Respond with ONLY the exact category name and nothing else.
- Do not add punctuation, quotes, or explanation.

Customer Query:
{query}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise classifier assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        predicted = response.choices[0].message.content.strip().strip('"').strip("'")
        for cat in INTENT_CATEGORIES:
            if cat.lower() in predicted.lower():
                return cat
        return "General / Other Support"
    except Exception:
        return "General / Other Support"
