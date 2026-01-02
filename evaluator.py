import streamlit as st
import os
import google.generativeai as genai
from sentence_transformers import util
from dotenv import load_dotenv

load_dotenv()


def calculate_similarity(model, prompt1, prompt2):
    """
    Calculates the cosine similarity between two prompts using a sentence-transformer model.
    """
    if not prompt1 or not prompt2:
        return 0.0

    try:
        embedding1 = model.encode(prompt1, convert_to_tensor=True)
        embedding2 = model.encode(prompt2, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        similarity_score = cosine_scores.item()

        return similarity_score
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0


REGENERATION_PROMPT_TEMPLATE = """
    You are an expert in prompt engineering. Your task is to revise a user's prompt based on their feedback and grounded examples.

    **Original Question:**
    {original_question}

    **User's Initial Prompt:**
    {user_prompt}

    **User's Feedback for Improvement:**
    {user_feedback}

    **Relevant Examples from our Dataset (for context):**
    ---
     **Example Prompt:** {example_prompt}                                                                                                 │
│    **Example Hints:** {example_hints}
    ---

    **Your Task:**
    Rewrite the user's initial prompt. Incorporate their feedback and adhere to the style and structure of the provided examples. The new prompt must be clear,in-depth-specific, and directly address the original question.
    produce best generted with no additional commentary.
    output format :

    '''
        return generated_prompt here.
    '''

"""


def regenerate_prompt_with_llm(
    original_question, user_prompt, user_feedback, example_prompt, example_hints
):
    """
    Uses the Gemini LLM to regenerate a prompt based on user feedback and a single direct example.
    """
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not configured. Cannot regenerate prompt.")
        return "Error: API key not configured."

    # Configure and call the LLM
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llm = genai.GenerativeModel("gemini-2.5-flash")

    formatted_prompt = REGENERATION_PROMPT_TEMPLATE.format(
        original_question=original_question,
        user_prompt=user_prompt,
        user_feedback=user_feedback,
        example_prompt=example_prompt,
        example_hints=example_hints,
    )

    try:
        response = llm.generate_content(formatted_prompt)
        regenerated_prompt = response.text
        return regenerated_prompt
    except Exception as e:
        st.error(f"Error during LLM regeneration: {e}")
        return f"Failed to regenerate prompt. Error: {e}"


def rag_evaluate(model, user_prompt, question, df):
    """
    Finds all model_prompts for a given question in the dataframe,
    calculates the similarity of the user_prompt against each of them,
    and returns the highest similarity score.
    """
    if not user_prompt or question not in df["question"].values:
        return 0.0

    # Find all reference prompts for the given question
    reference_prompts = df[df["question"] == question]["model_prompt"].tolist()

    if not reference_prompts:
        st.warning(f"No model prompts found for the question: {question}")
        return 0.0

    try:
        user_embedding = model.encode(user_prompt, convert_to_tensor=True)
        reference_embeddings = model.encode(reference_prompts, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(user_embedding, reference_embeddings)

        best_score = cosine_scores.max().item()

        return best_score
    except Exception as e:
        st.error(f"Error during RAG evaluation: {e}")
        return 0.0
