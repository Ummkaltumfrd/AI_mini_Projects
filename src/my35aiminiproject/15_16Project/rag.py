import os
from pathlib import Path

import chromadb
import ollama
from chunking_comparison import fixed_chunks, hybrid_chunks, sentence_chunks_result
from dotenv import load_dotenv

load_dotenv()
emb_model=os.environ.get("embb_model")
model=os.environ.get("ai_model")

pdf_path=Path(__file__).parent

# ==========================================
# ChromaDB
# ==========================================
"""    Project 15
chroma_client = chromadb.PersistentClient(
    path=str(pdf_path / "chroma_db")
)

collection = chroma_client.get_collection(
    name="document_collection"
) """

chroma_client = chromadb.PersistentClient(
    path=str(pdf_path / "chroma_db")
)

hybrid_collection = chroma_client.get_or_create_collection(
    name="hybrid_collection"
)

fixed_collection = chroma_client.get_or_create_collection(
    name="fixed_collection"
)

sentence_collection = chroma_client.get_or_create_collection(
    name="sentence_collection"
)

# ==========================================
# Retrieve relevant chunks
# ==========================================

def retrieve(qst,chunk_type,n_res=4):

  response =ollama.embed(
    model=emb_model,
    input=qst
  )

  question_embedding = response["embeddings"][0]

  results= chunk_type.query(
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

    hybrid_chunks = retrieve(question,hybrid_collection)
    fixed_chunks = retrieve(question,fixed_collection)
    sentence_chunks = retrieve(question,sentence_collection)

    print("HYBRID CHUNKS:", len(hybrid_chunks))
    print("FIXED CHUNKS:", len(fixed_chunks))
    print("SENTENCE CHUNKS:", len(sentence_chunks))

    print("\n--- HYBRID SAMPLE ---")
    print(hybrid_chunks[:2])

    print("\n--- FIXED SAMPLE ---")
    print(fixed_chunks[:2])

    print("\n--- SENTENCE SAMPLE ---")
    print(sentence_chunks[:2])

    context_parts = []

    for chunk in hybrid_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']['source']}

CONTENT:
{chunk['text']}
"""
        )

    for chunk in fixed_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']['source']}

CONTENT:
{chunk['text']}
"""
        )

    for chunk in sentence_chunks:

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

    return answer, hybrid_chunks,fixed_chunks,sentence_chunks

#===========================================
# Helper : add chunks to Chroma  
#===========================================

def add_chunks(collection,chunks,prefix):
   documents = []
   embeddings = []
   ids = []
   metadatas = []

   for i,text in enumerate(chunks):
      response = ollama.embed(
         model= emb_model,
         input =text
      )

      embedding = response["embeddings"][0]

      documents.append(text)
      embeddings.append(embedding)
      ids.append(f"{prefix}_{i}")

      metadatas.append({
         "source": pdf_path.name,
         "chunk_id":i,
         "chunk_type": prefix
      })
   collection.upsert(
      ids=ids,
      documents=documents,
      embeddings=embeddings,
      metadatas=metadatas
   )

   print(
      f"{prefix}: stored {len(documents)} chunks"
   )

#  ==========================================
# Store in ChromaDB
# ==========================================

add_chunks(
    hybrid_collection,
    hybrid_chunks,
    "hybrid"
)

add_chunks(
    fixed_collection,
    fixed_chunks,
    "fixed"
)

add_chunks(
    sentence_collection,
    sentence_chunks_result,
    "sentence"
)

# ==========================================
# Verify
# ==========================================

print("\n==============================")
print("CHROMA DB")
print("==============================")

print(
    "Hybrid:",
    hybrid_collection.count()
)

print(
    "Fixed:",
    fixed_collection.count()
)

print(
    "Sentence:",
    sentence_collection.count()
)