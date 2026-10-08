import os
from pathlib import Path

import chromadb
import ollama
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()
model =os.environ.get("embb_model")
BASE_PATH=Path(__file__).parent 
#--------------------------------------
#  Convert PDF with Docling
#--------------------------------------

print("Reading PDF with Docling...")

pdf_path =Path(__file__).parent  / "35_Projects.pdf"
converter =DocumentConverter()

result = converter.convert(pdf_path)

markdown= result.document.export_to_markdown()

print("PDF converted successfully.")

#--------------------------------------
#  Chunk the document
#--------------------------------------

def create_chunck(text,chunk_size=1000,overlap=200):
  chunks=[]
  start=0

  while start < len(text):
    end =start+chunk_size
    chunk= text[start:end].strip()

    if chunk:
      chunks.append(chunk)

    start+=chunk_size -overlap
  return chunks

chunks = create_chunck(markdown)

print(f"Created {len(chunks)} chunks.")

#--------------------------------------
#  Connect to ChromaDB
#--------------------------------------

chroma_client = chromadb.PersistentClient(
  path=str(BASE_PATH/ "chroma_db")
)

collection =chroma_client.get_or_create_collection(
  name="document_collection"
)

#--------------------------------------
#  Create embeddings and store chunks
#--------------------------------------
print("Creating embeddings...")

for i,chunk in enumerate(chunks):
  response = ollama.embed(
    model=model,
    input=chunk
  )

  embedding = response["embeddings"][0]

  collection.upsert(
    ids=[f"chunk_{i}"],
    embeddings=[embedding],
    documents=[chunk],
    metadatas=[
      {
        "source": pdf_path.name,
        "chunck_id": i
      }
    ]
  )
  print(f"Stored chunk {i + 1}/{len(chunks)}")

print("\nIngestion completed successfully!")
print(f"Stored {len(chunks)} chunks in ChromaDB.")