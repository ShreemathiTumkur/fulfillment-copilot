# scripts/rag_search.py
import faiss, numpy as np, pickle, argparse
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/shipments.faiss"
MAP_PATH   = "data/index/id_map.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(MAP_PATH, "rb") as f:
        id_map = pickle.load(f)          # list of dicts w/ 'delay_reason'
    return index, id_map

def embed(model, text: str) -> np.ndarray:
    vec = model.encode([text]).astype("float32")
    return vec

def search(query: str, k: int = 3):
    index, id_map = load_index()
    model = SentenceTransformer(MODEL_NAME)
    q_vec = embed(model, query)
    dist, idx = index.search(q_vec, k)
    results = [
        {"rank": r+1,
         "dist": float(d),
         "text": id_map[i]["delay_reason"][:160]}
        for r, (i, d) in enumerate(zip(idx[0], dist[0]))
    ]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="free-text query")
    parser.add_argument("-k", type=int, default=3, help="top-K")
    args = parser.parse_args()

    results = search(args.query, args.k)        # ← add this
    for r in results:                           # ← change loop header
        print(f"{r['rank']}. ({r['dist']:.2f}) {r['text']}")

    # ----- LLM answer -----

import os, textwrap
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

context = "\n".join(f"{r['rank']}. {r['text']}" for r in results)

prompt = textwrap.dedent(f"""
    You are Fulfillment-Copilot, a helpful logistics assistant.
    Answer the user's question only using the information in the context.
    If the context is not relevant, say "I don't have enough information."

    Context:
    {context}

    Question: {args.query}
    Answer in 2-3 sentences:
""").strip()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
)

answer = response.choices[0].message.content.strip()
print("\n---\nAnswer:\n", answer)

