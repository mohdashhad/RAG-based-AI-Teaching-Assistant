# import requests
# import os
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# jsons = os.listdir("jsons")  # List all the jsons 
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])
       
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk) 

# df = pd.DataFrame.from_records(my_dicts)
# # Save this dataframe
# joblib.dump(df, 'embeddings.joblib')




import requests
import os
import json
import pandas as pd
import joblib

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"


def create_embeddings(text_list):
    embeddings = []
    print(f"Sending {len(text_list)} chunks to Ollama for embeddings...")

    # for text in text_list:
    for text in text_list[:1]:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": text
            },
            timeout=120
        )
        data = response.json()

        if "error" in data:
            raise RuntimeError(f"Ollama Error: {data['error']}")

        # embeddings.append(data["data"][0]["embedding"])
        embeddings.append(data["embedding"])

    print(f"Received {len(embeddings)} embeddings")
    return embeddings


# -------------------- MAIN SCRIPT --------------------

print("\n========== RAG PREPROCESSING STARTED ==========\n")

json_folder = "jsons"
json_files = os.listdir(json_folder)

print(f"Found {len(json_files)} JSON files")

all_chunks = []
chunk_id = 0

for json_file in json_files:
    print(f"\nProcessing file: {json_file}")

    with open(os.path.join(json_folder, json_file), "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]
    print(f"Number of chunks in file: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)
    # print("CONTENT  CONTENT  : ", content)
    # print("CHUNKS CHUNKS : ", chunks)
    for i, chunk in enumerate(chunks[:len(embeddings)]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        all_chunks.append(chunk)

    print(f"Finished embedding {json_file}")

print("\nCreating DataFrame...")
df = pd.DataFrame.from_records(all_chunks)

print(f"DataFrame shape: {df.shape}")
print("Columns:", df.columns.tolist())

print("\nSaving embeddings to embeddings.joblib")
joblib.dump(df, "embeddings.joblib")

print("\n========== PREPROCESSING COMPLETED SUCCESSFULLY ==========\n")
