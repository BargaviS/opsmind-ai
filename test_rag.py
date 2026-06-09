"""
Quick smoke test — run with:
  PYTHONPATH=. python test_pipeline.py
"""
import sys

def test_chunker():
    from app.rag.chunker import DocumentChunker
    chunker = DocumentChunker()
    chunks = chunker.split("Hello world. " * 100)
    assert len(chunks) > 0, "No chunks produced"
    print(f"✅ Chunker: {len(chunks)} chunks produced")

def test_vector_store():
    from app.rag.vector_store import VectorStore
    store = VectorStore()

    store.add_chunks(
        document_id="test-doc-001",
        filename="test.txt",
        chunks=["OpsMind AI is a RAG system", "It indexes documents and answers questions"],
    )

    results = store.search("what does OpsMind do?", top_k=2)
    assert len(results) > 0, "No search results returned"
    print(f"✅ VectorStore: search returned {len(results)} results")
    for r in results:
        print(f"   score={r['score']} | {r['chunk'][:60]}")

def test_retriever():
    from app.rag.vector_store import get_vector_store
    from app.rag.retriever import Retriever

    store = get_vector_store()
    retriever = Retriever(store)
    chunks = retriever.retrieve("OpsMind", top_k=2)
    context = retriever.build_context(chunks)
    assert context, "Empty context"
    print(f"✅ Retriever: context built ({len(context)} chars)")

if __name__ == "__main__":
    print("Running OpsMind AI pipeline tests...\n")
    try:
        test_chunker()
        test_vector_store()
        test_retriever()
        print("\n✅ All tests passed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
