"""View what database has"""
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
  name="company_documents"
) 

print("total chunks: ",collection.count())

data = collection.get(
  include=["documents","metadatas"]
)

for i , (doc,meta) in enumerate(
  zip(data["documents"], data["metadatas"]),start=1
):
  print("\n" + "="* 50)
  print("Chunk:",i)
  print("Filename:",meta["filename"])
  print("Section:",meta["section"])
  print("Chunk index:",meta["chunk_index"])
  print("Text:",doc[:300])