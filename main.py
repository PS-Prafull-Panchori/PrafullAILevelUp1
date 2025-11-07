

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import gradio as gr


model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about SAP EHS.

Previous conversation (latest turns first):
{history}

Here are the most relevant review snippets (use them as context):
{reviews}

Now answer the user's question clearly and concisely. If the reviews do not contain the answer,
say "Not in provided context." and suggest what info is needed.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def _format_docs(docs, max_chars=300):
    """Make retriever outputs compact for the prompt and show short 'sources' list."""
    previews = []
    sources_view = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip().replace("\n", " ")
        snippet = (text[:max_chars] + "…") if len(text) > max_chars else text
        previews.append(f"- [{i}] {snippet}")
        meta_rating = d.metadata.get("rating", "")
        meta_date = d.metadata.get("date", "")
        sources_view.append(f"[{i}] rating={meta_rating} date={meta_date}")
    return "\n".join(previews), "\n".join(sources_view)

def _history_text(chat_history, keep_last=3):
    if not chat_history:
        return "—"
    # chat_history is List[List[str,str]] from Gradio: [[user, assistant], ...]
    last = chat_history[-keep_last:]
    lines = []
    for u, a in last[::-1]:  # newest first
        lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)

# --- Chat handler ---
def chat_fn(message, history):
    # Retrieve context
    docs = retriever.invoke(message)  # uses your Chroma retriever defined in vector.py
    reviews_text, sources_view = _format_docs(docs)

    # Build short conversation window for the model
    hist_text = _history_text(history, keep_last=3)

    # Generate
    result = chain.invoke({"reviews": reviews_text, "question": message, "history": hist_text})

    # Append a compact "Sources" block for transparency
    decorated = result
    if sources_view.strip():
        decorated = f"{result}\n\n---\nSources (top-k):\n{sources_view}"

    return decorated

# --- Launch simple chat UI ---
def main():
    gr.ChatInterface(
        fn=chat_fn,
        title="SAP EHS Agent (RAG, Ollama) Created by Prafull Panchori",
        description=(
            "Ask questions about SAP EHS. The assistant uses your local RAG index for context. "
            "If the answer isn't in the retrieved snippets, it will say 'Not in provided context.'"
        ),
        examples=[
            "Please click on this or type - What is SAP EHS"
            "What do the reviews say about Section 12.6 in SDS?",
            "Summarize feedback on identifiers and language handling.",
            "Any notes about Section 9 appearance suppression?"
        ],
    ).launch(share=True)

if __name__ == "__main__":
    main()
