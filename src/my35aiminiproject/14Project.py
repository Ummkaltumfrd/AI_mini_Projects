"""
THE 14TH PROJECT : Hybrid	Search	+	Reranking
Learn:
1.BM25 search -> to search by exct name 
2. Hybrid search ->  bm25 +  semantic
3. Reranking -> mechanism to rerank the results (for each document it will see how much a doc relevent to the query.) from the result of the Hybird search.(not nessecerly)
4.Dot Product (A . B) → measures similarity by taking into account both the direction and magnitude (length) of the vectors.

Cosine Similarity (A . B)/ A + B → measures similarity based only on the direction of the vectors, ignoring their magnitude.
   IT DEPENDS on the object of the model about who to use.

5. sentence-transforms by model = all-MiniLM-L6-v2-> use Dot Product for semantic search 

"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

documents = [
    "The new docking station provides fast charging, multiple USB ports, HDMI output, and reliable connectivity for laptops.",
    
    "The ZX-4817 is a USB-C docking station designed for connecting monitors, keyboards, mice, and external storage devices.",
    
    "This professional docking station supports dual monitors, Ethernet, USB 3.0, and power delivery for modern workstations.",
    
    "The USB-C hub is ideal for office workers who need additional ports and external display support.",
    
    "This compact laptop accessory offers HDMI, USB-A, Ethernet, and SD card connectivity in a portable design.",
    
    "The premium workstation dock provides 100W charging and support for multiple high-resolution displays.",
    
    "The ZX-4817 docking device was released as part of a new generation of computer accessories."
]

query = "ZX-4817"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(documents)
query_embedding = embedding_model.encode(query)

similarities = doc_embeddings @ query_embedding

semantic_results= sorted(
  enumerate(similarities),
  key=lambda x: x[1],
  reverse=True
)

print("\n=== Semantic Search ===")

for idx,score in semantic_results[:3]:
  print(score,documents[idx])

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)
tokenized_query =query.lower().split()

bm25_scores =bm25.get_scores(tokenized_query)

bm25_results = sorted(
  enumerate(bm25_scores),
  key=lambda x: x[1],
  reverse=True
)

print("\n=== BM25 Search ===")

for idx, score in bm25_results[:3]:
  print(score,documents[idx])

candidate_ids = set()

for idx, _ in semantic_results[:3]:
  candidate_ids.add(idx)

for idx, _ in bm25_results[:3]:
  candidate_ids.add(idx)

candidates = list(candidate_ids)

print("\n=== Hybrid Candidates ===")

for idx in candidates:
  print(documents[idx])

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
  (query,documents[idx])
  for idx in candidates
]

rerank_scores = reranker.predict(pairs)

reranked = sorted(
  zip(candidates,rerank_scores),
  key=lambda x: x[1],
  reverse=True
)

print("\n=== Final Reranked Results ===")

for  idx,score in reranked:
  print(score,documents[idx])