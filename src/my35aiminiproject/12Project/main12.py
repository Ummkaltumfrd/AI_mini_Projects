"""
THE 12TH PROJECT : Semantic Search Over Your Own Notes
1.semantic search (search by meaning using the embbending)
"""

import os

import numpy as np
import ollama
from dotenv import load_dotenv

load_dotenv()

model = os.environ.get("embb_model")
client = ollama.Client()


# ========================================
#              USER INPUT
# ========================================

input = [
    "What are some ways to improve my skills when preparing food?",
    "How can I build a healthier and more active routine?",
    "What can I do outdoors to stay active on my days off?",
    "How should I plan my vacation expenses?",
    "How can I get better at creating computer applications?"
]


# ========================================
#                FILES
# ========================================

folder = os.path.join(
    os.path.dirname(__file__),
    "TxtFiles"
)

files = []

for filename in os.listdir(folder):

    if filename.endswith(".txt"):

        filePath = os.path.join(folder, filename)

        with open(filePath, "r", encoding="utf-8") as f:

            files.append({
                "filename": filename,
                "text": f.read()
            })


# ========================================
#             FILE EMBEDDINGS
# ========================================

embb_files = []

for file in files:

    response = client.embed(
        model=model,
        input=file["text"]
    )

    embedding = response.embeddings[0]

    embb_files.append({
        "filename": file["filename"],
        "embedding": embedding
    })


# ========================================
#          INPUT EMBEDDING
# ========================================

response = client.embed(
    model=model,
    input=input[2]
)

text = response.embeddings[0]


print("Embedding length:", len(text))


# ========================================
#          COSINE SIMILARITY
# ========================================

results = []

for file in embb_files:

    similarity = np.dot(text, file["embedding"]) / (
        np.linalg.norm(file["embedding"]) *
        np.linalg.norm(text)
    )

    results.append({
        "filename": file["filename"],
        "score": similarity
    })


# ========================================
#                SORT
# ========================================

results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ========================================
#                RESULT
# ========================================

print("=" * 60)
print("THE RESULT")
print("=" * 60)

for result in results:

    print(
        result["filename"],
        ":",
        result["score"]
    )


# ========================================
#             BEST MATCH
# ========================================

print("=" * 60)

print(
    "The closest to the input is:",
    results[0]["filename"]
)

print(
    "Similarity:",
    results[0]["score"]
)

print("=" * 60)