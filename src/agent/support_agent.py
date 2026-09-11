import pandas as pd
from typing import Optional
from groq import Groq
from src.agent.llm_client import GROQ_MODEL
from src.retrieval.dense_retriever import retrieve_similar_cases


def generate_support_response(
    customer_query: str,
    embedding_model,
    index,
    rag_data: pd.DataFrame,
    client: Optional[Groq] = None,
    k: int = 5
) -> str:
    if client is None:
        return "Human support recommended: Groq API is not configured."

    similar_cases = retrieve_similar_cases(
        customer_query,
        embedding_model,
        index,
        rag_data,
        k=k
    )

    if similar_cases.empty:
        return "Human support recommended: No sufficiently similar historical case was found."

    evidence_blocks = []
    for _, row in similar_cases.iterrows():
        evidence_blocks.append(
            f"Customer issue:\n{row['customer_query']}\n\nHistorical Spotify response:\n{row['support_response']}\n"
        )
    evidence = "\n".join(evidence_blocks)

    prompt = f"""You are a customer support assistant for Spotify.

Draft ONE concise response to the customer's message.
Use ONLY the historical Spotify support examples provided below as evidence.

Rules:
1. Do not invent policies.
2. Do not invent refunds.
3. Do not invent troubleshooting steps.
4. Do not claim an action has been completed.
5. If historical evidence is insufficient, say that human support should review the case.
6. Do not mention AI.
7. Keep the response professional and concise.
8. Adapt the response to the customer's exact issue.
9. Return only the final customer-facing reply.
10. Do not provide multiple alternatives.

Customer message:
{customer_query}

Historical support evidence:
{evidence}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful Spotify customer support assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_text = str(e)
        if "429" in error_text:
            return "Human support recommended: the language model API rate limit was reached."
        return "Human support recommended: the response generation service was unavailable."
