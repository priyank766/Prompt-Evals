import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@st.cache_data
def create_faiss_index(_df, _model):
    if "question" not in _df.columns:
        st.error("DataFrame must contain a 'question' column.")
        return None, None

    questions = _df["question"].astype(str).tolist()

    try:
        embeddings = _model.encode(questions, convert_to_tensor=False)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index, embeddings
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        return None, None


def get_hints_from_question(user_question, df, model, index, k=3):
    """
    Retrieves the top k hints for a given user question using FAISS similarity search.
    """
    if index is None:
        return [], []

    try:
        question_embedding = model.encode([user_question])
        faiss.normalize_L2(question_embedding)

        # D: distances (inner product), I: indices of the nearest neighbors
        D, I = index.search(question_embedding, k)

        retrieved_hints = df.iloc[I[0]]["hints"].tolist()
        retrieved_rows = df.iloc[I[0]]

        return retrieved_hints, retrieved_rows
    except Exception as e:
        st.error(f"Error during hint retrieval: {e}")
        return [], []
