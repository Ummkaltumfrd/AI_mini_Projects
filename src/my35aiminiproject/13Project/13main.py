"""
THE 13TH PROJECT : Add	a	Real	Vector	Database
Learn:
1.use a vector database chromadb

"""
import os
import time

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

model=os.environ.get("embb_model")
#======================
# ChromaDB
#======================
client =chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
  name="documents"
)

#======================
# Ollama embedding
#======================
def get_embedding(text):
  response = ollama.embeddings(
    model=model,
    prompt=text
  )
  return response["embedding"]

#=======================
# Add one document
#=======================
def add_document(doc_id,text):
  embedding = get_embedding(text)

  collection.add(
    ids=[doc_id],
    documents=[text],
    embeddings=[embedding]
  )

#=======================
# delete one document
#=======================
def delete_document(doc_id):
 collection.delete(ids=[doc_id])

#=======================
# Search
#=======================

def search(query,n_results=5):
  embedding =get_embedding(query)

  return collection.query(
    query_embeddings=[embedding],
    n_results=n_results
  )

#========================
# Benchmark
#========================

def benchmark(number_of_docs):

  # Add documents
  for i in range(number_of_docs):
    doc_id = f"doc_{number_of_docs}_{i}"

    text=(
      f"This is document number {i}. "
      f"It contains information about artificial intelligence, "
      f"machine learning and vector databases."
    )

    add_document(doc_id,text)
 # Query benchmark
    query_embedding = get_embedding(
        "information about artificial intelligence"
    )

    start = time.perf_counter()

    collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    end = time.perf_counter()

    elapsed = end - start

    print(
        f"{number_of_docs} documents -> "
        f"{elapsed * 1000:.3f} ms"
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    print("Benchmark:")

    benchmark(100)
    benchmark(1000)

    # Test adding ONE document
    print("\nAdding one document...")

    start = time.perf_counter()

    add_document(
        "single_test",
        "This is a newly added document."
    )

    end = time.perf_counter()

    print(
        f"Add time: {(end - start) * 1000:.3f} ms"
    )

    # Test deleting ONE document
    print("\nDeleting one document...")

    start = time.perf_counter()

    delete_document("single_test")

    end = time.perf_counter()

    print(
        f"Delete time: {(end - start) * 1000:.3f} ms"
    )

    # Test search
    print("\nSearch results:")

    results = search(
        "machine learning"
    )

    print(results["documents"])