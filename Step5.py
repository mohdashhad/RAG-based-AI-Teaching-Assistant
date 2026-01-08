import joblib
import numpy as np
import requests

# -------------------------------
# Load embeddings
# -------------------------------
print("\nLoading stored embeddings...")
df = joblib.load("embeddings.joblib")

print("Embeddings loaded successfully")
print("Total chunks:", len(df))

# -------------------------------
# Ollama config
# -------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "bge-m3"
# LLM_MODEL = "llama3"
LLM_MODEL = "llama3:8b"


# -------------------------------
# Embed user query
# -------------------------------
def embed_query(query):
    print("\nCreating embedding for query...")
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": query
        }
    )

    data = response.json()

    if "error" in data:
        raise RuntimeError("Embedding error: " + data["error"])

    print("Query embedding created")
    return np.array(data["embedding"])

# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve_relevant_chunks(query, df, top_k=5):
    print("\nRetrieving relevant chunks...")
    query_embedding = embed_query(query)

    similarities = []
    for _, row in df.iterrows():
        score = cosine_similarity(
            query_embedding,
            np.array(row["embedding"])
        )
        similarities.append(score)

    df = df.copy()
    df["similarity"] = similarities

    top_chunks = df.sort_values("similarity", ascending=False).head(top_k)

    print(f"Top {top_k} chunks retrieved")
    return top_chunks

# -------------------------------
# Build prompt
# -------------------------------
def build_prompt(query, retrieved_chunks):
    print("\nBuilding prompt for LLM...")
    context = "\n\n".join(
        f"- {row['text']}"
        for _, row in retrieved_chunks.iterrows()
    )

    prompt = f"""
You are a helpful assistant.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt

# -------------------------------
# Ask LLM safely
# -------------------------------
# def ask_llm(prompt):
#     print("\nSending prompt to LLM...")
#     response = requests.post(
#         OLLAMA_GEN_URL,
#         json={
#             "model": LLM_MODEL,
#             "prompt": prompt,
#             "stream": False
#         }
#     )

#     data = response.json()

#     print("Raw LLM response:")
#     print("RAW LLM RESPONSE DATA data : ", data)

#     if "error" in data:
#         raise RuntimeError("LLM error: " + data["error"])

#     if "response" not in data:
#         raise RuntimeError("Unexpected LLM response format")

#     return data["response"]


def ask_llm(prompt):
    print("\nSending prompt to LLM (CPU mode)...")

    response = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_gpu": 0   # 🔥 FORCE CPU
            }
        }
    )

    data = response.json()

    print("Raw LLM response:")
    print(data)

    if "error" in data:
        raise RuntimeError("LLM error: " + data["error"])

    if "response" not in data:
        raise RuntimeError("Unexpected LLM response format")

    return data["response"]



# -------------------------------
# MAIN
# -------------------------------
query = "What is VS Code and why is it used?"

top_chunks = retrieve_relevant_chunks(query, df, top_k=5)
prompt = build_prompt(query, top_chunks)

answer = ask_llm(prompt)

print("\n========== FINAL ANSWER ==========\n")
print(answer)
