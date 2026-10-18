import hashlib
import os
import re
from pathlib import Path

import chromadb
import ollama
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR =Path("multi_document_rag_10_docs")
DB_DIR =Path("chroma_db")
COLLECTION_NAME ="company_documents"
EMBED_MODEL = os.environ.get("embb_model")
SUPPORTED_EXTENSIONS = {".pdf",".docx",".pptx",".txt"}

converter = DocumentConverter()

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
  name=COLLECTION_NAME,
  metadata={"hnsw:space": "cosine"}
)

def read_document(path:Path) -> str:
  if path.suffix.lower() ==".txt":
    return path.read_text(encoding="utf-8")
  result = converter.convert(str(path))
  return result.document.export_to_markdown()

def split_into_sections(text:str):
  """
  Split Markdown by headings.
  Text before the first heading belongs to General
  """
  sections=[]
  current_title = "General"
  current_lines = []

  for line in text.splitlines():
    match =re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$",line)

    if match:
      content = "\n".join(current_lines).strip()

      if content:
        sections.append((current_title,content))

      current_title = match.group(1).strip()
      current_lines=[]
    else:
      current_lines.append(line)

  content= "\n".join(current_lines).strip()
  if content:
    sections.append(("General",text.strip()))
  return sections

def chunk_text(text:str,chunk_size=1200,overlap=200):
  """Craete overlaping character-based chunks."""
  text = re.sub(r"\n{3,}","\n\n",text).strip()
  chunks = []
  start = 0

  while start < len(text):
    end = min(start+ chunk_size,len(text))

    if end < len(text):
      boundary =text.rfind("\n", start+chunk_size//2,end)
      if boundary > start:
        end =boundary

    chunk =text[start:end].strip()
    if chunk:
      chunks.append(chunk)

    if end >= len(text):
      break

    start = max(end - overlap , start + 1)
  return chunks

def embed_text(text: str):
  response = ollama.embed(
    model=EMBED_MODEL,
    input=text
  )
  return response["embeddings"][0]

def main():
  if not DOCS_DIR.exists():
    raise FileNotFoundError(
      "Create the documents/ folder and put your 10 files inside it"
    )

  files = sorted(
    p for p in DOCS_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
  )

  if not files:
    raise RuntimeError("No supported documents found in documents/.")

  ids,documents,metadatas,embeddings = [],[],[],[]
  seen_files = set()

  for path in files:
    print(f"Reading:{path.name}")

    try:
       text = read_document(path)
       if not text.strip():
         print (f"Skipped empty document:{path.name}")
         continue

       sections = split_into_sections(text)
       file_type = path.suffix.lower().lstrip(".")

       for section_name ,section_text in sections:
         for chunk_index ,chunk in enumerate(chunk_text(section_text)):
           raw_id = (
             f"{path.name}|{section_name}"
             f"{chunk_index}|{chunk}"
           )
           chunk_id = hashlib.sha256(
             raw_id.encode("utf-8")
           ).hexdigest()

           ids.append(chunk_id)
           documents.append(chunk)
           metadatas.append({
             "filename": path.name,
             "section": section_name,
             "file_type": file_type,
             "chunk_index": chunk_index,
           })
           embeddings.append(embed_text(chunk))
       seen_files.add(path.name)
       print(f" Extracted sections:{len(sections)}")     
    except Exception as exc:
      print(f" ERROR processing {path.name}: {exc}")

  if not ids:
   raise RuntimeError("No chunks were create. Check your files and logs.")
# Upsert makes re-running ingestion safe for unchanged chunk IDs.
  batch_size = 50
  for start in range(0,len(ids),batch_size):
    end = start + batch_size
    collection.upsert(
      ids=ids[start:end],
      documents=documents[start:end],
      metadatas=metadatas[start:end],
      embeddings=embeddings[start:end],
    )

  print("\n Ingestion complete.")
  print(f"Files processed: {len(seen_files)}")
  print(f"Chunks indexed: {len(ids)}")
  print(f"Collection: {COLLECTION_NAME}")
  print(f"Stored database: {DB_DIR.resolve()}")

if __name__ == "__main__":
  main()