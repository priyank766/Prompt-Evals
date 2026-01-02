import streamlit as st
import pandas as pd
from data_loader import load_data
from hint_engine import (
    load_embedding_model,
    create_faiss_index,
)
from evaluator import calculate_similarity, regenerate_prompt_with_llm, rag_evaluate

# --- Page Configuration ---
st.set_page_config(page_title="Finance Prompt Evaluator", page_icon="🤖", layout="wide")

# --- Load Data and Models (Cached) ---
df = load_data()
if df is not None:
    embedding_model = load_embedding_model()
    faiss_index, embeddings = create_faiss_index(df, embedding_model)
else:
    st.error("Failed to load the dataset. The application cannot start.")
    st.stop()

# --- UI Layout ---
st.title("Finance Prompt Evaluator 🏦")
st.write("Select a domain, topic, and question to start, then write your own prompt.")

# --- Session State Initialization ---
if "evaluation_score" not in st.session_state:
    st.session_state.evaluation_score = None
if "evaluation_type" not in st.session_state:
    st.session_state.evaluation_type = "Prompt Similarity Score"


# --- Sidebar for Selections ---
st.sidebar.header("1. Select Domain and Topic")
sub_domain_list = df["sub_domain"].unique().tolist()
selected_sub_domain = st.sidebar.selectbox("Sub-Domain", sub_domain_list)

topic_area_list = (
    df[df["sub_domain"] == selected_sub_domain]["topic_area"].unique().tolist()
)
selected_topic_area = st.sidebar.selectbox("Topic Area", topic_area_list)

# --- Main Panel for Prompting and Evaluation ---
st.header("Select Question, Write and Evaluate Your Prompt")

question_list = df[
    (df["sub_domain"] == selected_sub_domain)
    & (df["topic_area"] == selected_topic_area)
]["question"].tolist()
selected_question = st.selectbox("Select a Question", question_list)

# Get the full record for the selected question
selected_record = df[df["question"] == selected_question].iloc[0]

user_prompt = st.text_area(
    "Your Prompt:",
    height=150,
    placeholder="Describe the task for the AI based on the selected question...",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Evaluate Prompt", use_container_width=True, type="primary"):
        if user_prompt:
            model_prompt = selected_record["model_prompt"]
            similarity_score = calculate_similarity(
                embedding_model, user_prompt, model_prompt
            )

            if similarity_score < 0.65:
                st.info(
                    "Initial similarity is low. Performing a deeper evaluation against the dataset..."
                )
                rag_score = rag_evaluate(
                    embedding_model, user_prompt, selected_question, df
                )
                st.session_state.evaluation_score = rag_score
                st.session_state.evaluation_type = "RAG Score"
            else:
                st.session_state.evaluation_score = similarity_score
                st.session_state.evaluation_type = "Prompt Similarity Score"
        else:
            st.warning("Please write a prompt before evaluating.")

with col2:
    with st.expander("Get a Hint ✨"):
        st.info(selected_record["hints"])


if st.session_state.evaluation_score is not None:
    st.subheader("Evaluation Score")
    label = st.session_state.get('evaluation_type', 'Prompt Similarity Score')
    st.metric(
        label=label,
        value=f"{st.session_state.evaluation_score:.2%}",
    )
    st.progress(st.session_state.evaluation_score)

    with st.expander("Show Reference Model Prompt"):
        st.success(selected_record["model_prompt"])

    # --- Regeneration Section ---
    st.header(" Regenerate ")
    st.write(
        "If you're not satisfied, provide feedback and let the AI improve your prompt."
    )

    user_feedback = st.text_input(
        "Your Feedback for Regeneration:",
        placeholder="e.g., 'Make the output a JSON object', 'Be more concise'",
    )

    if st.button("Regenerate Prompt", use_container_width=True):
        if user_feedback:
            with st.spinner("🤖 The AI is thinking..."):
                regenerated_prompt = regenerate_prompt_with_llm(
                    original_question=selected_question,
                    user_prompt=user_prompt,
                    user_feedback=user_feedback,
                    example_prompt=selected_record["model_prompt"],
                    example_hints=selected_record["hints"],
                )
                st.subheader("AI-Regenerated Prompt")
                st.markdown(regenerated_prompt)
        else:
            st.warning("Please provide feedback for regeneration.")
