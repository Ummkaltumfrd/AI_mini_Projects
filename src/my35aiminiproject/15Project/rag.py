import os
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()
emb_model=os.environ.get("embb_model")
model=os.environ.get("ai_model")

pdf_path=Path(__file__).parent

# ==========================================
# ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path=str(pdf_path / "chroma_db")
)

collection = chroma_client.get_collection(
    name="document_collection"
)

# ==========================================
# Retrieve relevant chunks
# ==========================================

def retrieve(qst,n_res=4):

  response =ollama.embed(
    model=emb_model,
    input=qst
  )

  question_embedding = response["embeddings"][0]

  results= collection.query(
    query_embeddings=[question_embedding],
    n_results=n_res
  )

  documents = results["documents"][0]
  metadatas = results["metadatas"][0]
  distances = results["distances"][0]

  retrieved_chunks = []

  for document, metadata , distance in zip(
    documents,
    metadatas,
    distances
  ):
    retrieved_chunks.append(
      {
            "text": document,
            "source": metadata,
            "distance": distance
      }
    )
  return retrieved_chunks

# ==========================================
# Generate grounded answer
# ==========================================

def answer_question(question):

    chunks = retrieve(question)

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']['source']}

CONTENT:
{chunk['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

You MUST answer using ONLY the information contained
in the provided context.

If the context does not contain enough information
to answer the question, say exactly:

"I don't know based on the document."

Do NOT use your own knowledge.

Question:
{question}

Context:
{context}

Rules:
1. Use only the context.
2. Do not invent information.
3. If the answer is not supported, say:
   "I don't know based on the document."
4. Always provide the source chunk used.

Answer:
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer, chunks