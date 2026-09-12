import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_or_load_rag_index(rag_data: pd.DataFrame, embedding_model: SentenceTransformer, embedding_file: Path, faiss_file: Path):
    if faiss_file.exists():
        embedding_matrix = np.load(embedding_file) if embedding_file.exists() else None
        index = faiss.read_index(str(faiss_file))
        return embedding_matrix, index

    embeddings = embedding_model.encode(
        rag_data["customer_query"].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    embedding_matrix = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    embedding_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_file, embedding_matrix)
    faiss.write_index(index, str(faiss_file))

    return embedding_matrix, index


def retrieve_similar_cases(query: str, embedding_model: SentenceTransformer, index, rag_data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    query_embedding = embedding_model.encode([query])
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)
    valid_indices = [i for i in indices[0] if 0 <= i < len(rag_data)]

    results = rag_data.iloc[valid_indices].copy()
    results["distance"] = distances[0][:len(valid_indices)]
    return results
