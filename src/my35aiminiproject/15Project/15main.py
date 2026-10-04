"""
THE 15TH PROJECT : Basic	RAG
Learn:
1.RAG -> retrive data to use them (fresh and private data without needing to fine tune)
2. Retrieval -> transform the input into vectors and retive the data (from vector data base) and apply semantic search
 Augmention ->  the proccess where the retrived data is ingacted into the prent at the run time.(improve)
 Generation -> ai assistent generate the response given the semantic relevent data from the vector database 

3. convert pdf to a readable thing (dockling)-> chunking -> embedding -> chromadb (ingest.py)
4. user Qst -> retrive relvant chunks -> genertae grounded answer(prmpting)
"""

from rag import answer_question

print("=" * 60)
print("PDF RAG QUESTION ANSWERING SYSTEM")
print("=" * 60)

print("Type 'exit' to quit.\n")


while True:

    question = input("Question: ")

    if question.lower() == "exit":
        break

    if not question.strip():
        continue

    answer, chunks = answer_question(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for chunk in chunks:

        print(
            f"\nChunk {chunk['source']}"
            f" | Distance: {chunk['distance']:.4f}"
        )

        print(chunk["text"][:500])

    print()