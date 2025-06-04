import gradio as gr
import requests
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain.embeddings import HuggingFaceEmbeddings

# Load embedding model and vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorDB = FAISS.load_local("VDb", embedding_model, allow_dangerous_deserialization=True)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# General conversation checker
def is_general_conversation(query):
    greetings = [
        "hi", "hello", "how are you", "what's up", "hey", "good morning",
        "good evening", "who are you", "how do you work", "tell me about yourself"
    ]
    return any(greet in query.lower() for greet in greetings)

# Ollama call
def run_ollama(prompt, model="mistral", temperature=0.7):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
    )
    return response.json()["response"]

# Main function for CivilAI query, now returns (answer, doc_info)
def civilai_query_with_docs(query):
    if is_general_conversation(query):
        answer = (
            "👋 Hi! I'm CivilAI, your assistant for road-related documentation.\n\n"
            "📂 I help you understand forest clearance, NH projects, road widening, DPRs, etc.\n"
            "💬 Try asking things like:\n"
            " - What's the forest clearance status?\n"
            " - Details of NH-458 DPR?\n"
            " - What are the coordinates for section 2?\n"
        )
        doc_info = "No specific documents used for general conversation."
        return answer, doc_info

    # Step 1: Retrieve documents
    docs = vectorDB.similarity_search(query, k=10)
    if not docs:
        return "❌ Sorry, I couldn't find any relevant information in the documents.", "No documents found."

    # Step 2: Rerank
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    # Step 3: Sort and pick top 3
    top_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)][:3]

    # Step 4: Combine context (for prompt)
    combined_context = "\n\n".join(doc.page_content[:1000] for doc in top_docs)

    # Prepare document info with SOURCES
    doc_info = "\n\n**Documents used for this answer:**\n" + "\n\n---\n".join(
        f"**Document {i+1}** (Source: `{doc.metadata.get('source', 'Unknown')}`):\n"
        f"{doc.page_content[:300]}..." 
        for i, doc in enumerate(top_docs)
    )

    # ✅ Step 5: Prepare full prompt
    prompt = f"""You are a helpful assistant for CivilAI. Use the context below to answer clearly and precisely.

Context:
{combined_context}

Question: {query}

Answer:"""

    # Step 6: Get answer
    answer = run_ollama(prompt)
    return answer, doc_info

# Gradio UI with sources panel
with gr.Blocks(title="CivilAI Assistant", theme="default") as demo:
    gr.Markdown("# CivilAI Assistant")
    gr.Markdown("Ask questions about NH projects, road widening, forest clearance, and more.")

    with gr.Row():
        query = gr.Textbox(label="Your question", placeholder="Ask something like 'Tell me about NH-458 widening'...")
        submit_btn = gr.Button("Submit")

    answer = gr.Textbox(label="Answer", interactive=False)
    docs_used = gr.Markdown(label="Document Sources Used")

    def respond(query):
        answer, doc_info = civilai_query_with_docs(query)
        return answer, doc_info

    submit_btn.click(fn=respond, inputs=query, outputs=[answer, docs_used])

# Launch app
demo.launch()
