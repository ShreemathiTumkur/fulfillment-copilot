import streamlit as st
import faiss, numpy as np, pickle, pathlib, os, textwrap
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------------------------------------------------
#  Configuration
# -------------------------------------------------------------------
INDEX_PATH = "data/index/shipments.faiss"
MAP_PATH   = "data/index/id_map.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"   # fallback to gpt-3.5-turbo if needed
TOP_K = 5

# -------------------------------------------------------------------
#  Load artifacts (cached so they load only once)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FAISS index…")
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(MAP_PATH, "rb") as f:
        id_map = pickle.load(f)
    return index, id_map

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

index, id_map = load_index()
embedder = load_embedder()
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------
#  Helper: retrieve top-K passages
# -------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K):
    q_vec = embedder.encode([query]).astype("float32")
    dist, idx = index.search(q_vec, k)
    passages = [
        {"rank": r+1, "dist": float(d), "text": id_map[i]["delay_reason"]}
        for r, (i, d) in enumerate(zip(idx[0], dist[0]))
    ]
    return passages

# -------------------------------------------------------------------
#  Helper: call LLM with keyword-first prompt
# -------------------------------------------------------------------
def generate_answer(query: str, passages):
    context = "\n".join(f"{p['rank']}. {p['text']}" for p in passages)

    system_prompt = (
        "You are Fulfillment-Copilot, an expert logistics assistant. "
        "Begin your answer with exactly ONE word — Weather, Traffic, or Inventory — "
        "that best explains the delay, then a brief explanation. "
        "If none apply, respond exactly with 'Unknown'. "
        "Use ONLY the context."
    )

    prompt = textwrap.dedent(f"""
        {system_prompt}

        Context:
        {context}

        Question: {query}
        Answer (2-3 sentences):
    """).strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------------------------------------
#  Streamlit UI
# -------------------------------------------------------------------
st.title("Fulfillment-Copilot")

query = st.text_input("Ask why a shipment was delayed:", placeholder="e.g. Why was my shipment delayed?")
if st.button("Explain") or query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not set. `export OPENAI_API_KEY=...` in your Terminal and restart.")
    elif query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context & generating answer…"):
            passages = retrieve(query)
            answer   = generate_answer(query, passages)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Top-K context")
        for p in passages:
            st.write(f"**{p['rank']}.** {p['text']}  _(dist {p['dist']:.2f})_")

