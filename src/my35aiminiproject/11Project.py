"""
THE 11TH PROJECT : Embedding	Pipeline
Learn:
1.Embeddings (transform texts to number <vectors>).
2.Cosine similarity (tells us how similar two pieces of text are).

"""

import os

import numpy as np
import ollama
from dotenv import load_dotenv

load_dotenv()

client=ollama.Client()
model=os.environ.get('embb_model')

text ="Python is a programming language."
long_text="""
Artificial intelligence has become an important area of computer science.
Machine learning allows computers to learn patterns from data without being
explicitly programmed for every possible situation. One important technique
in modern AI systems is the use of embeddings. An embedding represents text
as a numerical vector. Texts with similar meanings tend to have vectors that
are close to each other in the embedding space.

Embeddings are useful in many applications, including semantic search,
recommendation systems, document classification, question answering, and
retrieval augmented generation. Instead of comparing documents only by the
words they contain, an embedding model can capture relationships between
different words and concepts. For example, the words dog and puppy may be
different strings, but their meanings are closely related.

A local embedding model such as nomic-embed-text can transform a sentence,
paragraph, or larger document into a fixed-size numerical representation.
This representation can then be compared with other vectors using measures
such as cosine similarity. A high cosine similarity generally indicates that
two pieces of text point in similar directions in the embedding space.

This makes embeddings particularly useful when working with large collections
of documents. A system can convert documents into vectors, store those
vectors, and later compare a user's query against them to find documents
that are semantically relevant.
"""

response = client.embed(
  model=model,
  input=long_text
)

print("text:",long_text)
print("the vector:",response.embeddings)

#==================================
#    cosine similarity
#==================================
pairs=[ 
 "I have a dog.",
 "I own a puppy.",
 
 "I like programming.",
 "I enjoy writing software.",
 
 "I have a dog.",
 "The weather is sunny today."
     ]

abc=[]
for pair in pairs:
 response = client.embed(
  model=model,
  input=pair
 )
 embedding=response.embeddings[0]
 abc.append(embedding)

a1=abc[0]
a2=abc[1]

b1=abc[2]
b2=abc[3]

c1=abc[4]
c2=abc[5]

sml1= np.dot(a1,a2)/np.linalg.norm(a1) * np.linalg.norm(a2)
sml2= np.dot(b1,b2)/np.linalg.norm(b1) * np.linalg.norm(b2)
sml3= np.dot(c1,c2)/np.linalg.norm(c1) * np.linalg.norm(c2)

print("the cosine similarity between a1 /a2:",sml1)
print("the cosine similarity between b1 /b2:",sml2)
print("the cosine similarity between c1 /c2:",sml3)