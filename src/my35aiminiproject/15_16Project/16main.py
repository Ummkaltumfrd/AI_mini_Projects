"""
THE 16TH PROJECT : Chunking	Strategy	Comparison
Learn:
1.compare 3 chuncking methods:
  1.hybird chunking(best for structred documents)
  2.sentence chunking
  3.fixed chunking

2.  the flow => transform pdf -> define the 3 chunking methods -> chunk the pdf -> create collection for each chunk method(chromadb) -> add the chunks to chromadb (embedded by ollama) -> loop over 10 questions -> call answer generate -> embed the questions -> retrieve data from chromadb -> generate an answer 

"""

from rag import answer_question

print("=" * 60)
print("PDF RAG QUESTION ANSWERING SYSTEM")
print("=" * 60)

print("Type 'exit' to quit.\n")

questions = [
    "What are the three chunking approaches compared in Project 16?",
    "What does chunking do before documents are embedded and stored?",
    "Why does cutting a table or sentence mid-way cause a problem?",
    "What is the main advantage of Docling's structure-aware HybridChunker?",
    "What type of document structure does HybridChunker preserve?",
    "What should you log when comparing the three chunking strategies?",
    "What should you write after comparing the three chunking strategies?",
    "Why should HybridChunker generally win on documents with real structure?",
    "What should you try if all three chunking strategies get the same score?",
    "What tools are recommended for Project 16?"
]

for question in questions:
    answer, hybrid_chunks,fixed_chunks,sentence_chunks = answer_question(question)

    print("\n" + "=" * 60)
    print("Q: ",question)
    print("=" * 60)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("HYBRID CHUNKS")
    print("=" * 60)

    for chunk in hybrid_chunks:

        print(
            f"\nChunk {chunk['source']}"
            f" | Distance: {chunk['distance']:.4f}"
        )

        print(chunk["text"][:500])

    print()

    print("\n" + "=" * 60)
    print("FIXED CHUNKS")
    print("=" * 60)

    for chunk in fixed_chunks:

        print(
            f"\nChunk {chunk['source']}"
            f" | Distance: {chunk['distance']:.4f}"
        )

        print(chunk["text"][:500])

    print()


    print("\n" + "=" * 60)
    print("SENTENCE CHUNKS")
    print("=" * 60)

    for chunk in  sentence_chunks:

        print(
            f"\nChunk {chunk['source']}"
            f" | Distance: {chunk['distance']:.4f}"
        )

        print(chunk["text"][:500])

    print()