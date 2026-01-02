# **Finance Prompt Evaluator for Domain Specific **

## **Overview**

A project that evaluates and improves AI prompts for the **finance and banking sector**. 

This system has a two-part workflow:
1.  **Initial Evaluation:** Users select a question and write a prompt. The system provides a basic similarity score by comparing the user's prompt to a reference prompt from the dataset using `sentence-transformers`.
2.  **Advanced Regeneration:** After the initial evaluation, users can opt to use an LLM-powered regeneration feature. This feature uses a **RAG (Retrieval-Augmented Generation)** pipeline over the project's dataset to enhance the prompt based on user feedback.

Built using **Streamlit**, **sentence-transformers**, and **Google Gemini** for the regeneration step.

---

## **Workflow**

### **1. User Input**

*   User selects from dropdown menus:
    *   `sub_domain` (e.g., Risk & Compliance, Retail Banking)
    *   `topic_area` (e.g., AML, KYC, Loan Risk)
    *   A `question` from a list filtered by the chosen domain and topic.

### **2. User Writes Prompt**

*   Based on the selected question, the user writes their own prompt in a text area.

### **3. Hint System (Optional)**

*   A **"Get Hint"** button displays the `hints` associated with the selected question from the dataset.

### **4. Prompt Evaluation (Sentence Similarity)**

*   The user's prompt is evaluated against the `model_prompt` from the dataset for the selected question.
*   **Technique:** The system calculates the cosine similarity between the user's prompt and the reference `model_prompt` using `sentence-transformers` to produce a simple score.

### **5. Display Model Prompt**

*   A **"Show Model Prompt"** button reveals the ideal `model_prompt` from the dataset, allowing the user to compare.

### **6. Prompt Regeneration (LLM + RAG)**

*   After the initial evaluation, a **"Regenerate"** button becomes available.
*   If the user is not satisfied, they can provide feedback (e.g., "make it produce JSON", "be more concise") and click "Regenerate".
*   The system uses an **LLM (Gemini) with a RAG pipeline** built on the `main-dataset.csv` to generate a new, improved prompt based on the original question, the user's initial prompt, and their feedback.

---

## **System Components**

| Component | Description | Tools / Libraries |
| --- | --- | --- |
| UI | Simple interface for selections, text input, and displaying results. | Streamlit |
| Data Loader | Loads the `main-dataset.csv` into memory. | Pandas |
| Hint Engine | Retrieves hints for the selected question from the dataset. The vector store and embeddings are cached after the first run. | sentence-transformers, FAISS |
| Prompt Evaluator | **1. Similarity:** Compares user prompt to a reference prompt. <br> **2. Regeneration:** Uses LLM+RAG for improvements. | sentence-transformers, Google Gemini |
| RAG Pipeline | Retrieves relevant examples (`model_prompt`, `hints`) from the dataset to ground the LLM for the regeneration step. | Custom/LangChain |
| Logging | Stores user interactions, prompts, scores, and feedback. | CSV / SQLite |

**Note on Efficiency:** The vector store (FAISS index) and sentence-transformer model are loaded only once and cached in memory (e.g., using `st.cache_resource`) to ensure the application is fast and avoids re-computing embeddings on every user interaction.

---

## **Scoring**

| Metric | Meaning | Range |
| --- | --- | --- |
| Similarity Score | How semantically similar the user's prompt is to the reference `model_prompt`. | 0% - 100% |

---

## **Tools Summary**

*   **UI:** Streamlit
*   **ML/NLP:** sentence-transformers, FAISS
*   **LLM:** Google Gemini (for regeneration)
*   **Database:** CSV (for dataset and logging)
*   **Versioning:** GitHub for dataset & logs
