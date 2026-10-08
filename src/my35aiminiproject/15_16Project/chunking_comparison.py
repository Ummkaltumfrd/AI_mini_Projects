import os
import re
from pathlib import Path

import ollama
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_PATH =BASE_DIR / "35_Projects.pdf"
emb_model =os.environ.get("embb_model")
# ==========================================
# Read PDF
# ==========================================

print("Reading PDF...")

converter = DocumentConverter()
result = converter.convert(PDF_PATH)

document = result.document

print("PDF converted successfully.")

markdown = document.export_to_markdown()

# ==========================================
# 1. HybridChunker
# ==========================================

hybrid_chunker = HybridChunker()
hybrid_doc_chunks = list(hybrid_chunker.chunk(document))

hybrid_chunks = [
    chunk.text
    for chunk in hybrid_doc_chunks
]
print(f"HybridChunker: {len(hybrid_chunks)} chunks")

# ==========================================
# 2. Fixed-size
# ==========================================

def fixed_size_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


fixed_chunks = fixed_size_chunks(markdown)

print(f"Fixed-size: {len(fixed_chunks)} chunks")

# ==========================================
# 3. Sentence-based
# ==========================================

def setence_chunks(text, sentences_per_chunk=5):
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    chunks = []

    for i in range(
        0,
        len(sentences),
        sentences_per_chunk
    ):
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        ).strip()

        if chunk:
            chunks.append(chunk)
    return chunks

sentence_chunks_result = setence_chunks(markdown)

print(
    f"Sentence-based: "
    f"{len(sentence_chunks_result)} chunks"
)

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
         "source": PDF_PATH.name,
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
