from app.rag.vector_store import VectorStore

store = VectorStore()

chunks = [
    "Project Alpha deadline is July 15",
    "Backend development completed",
    "Testing starts next week"
]

store.add_chunks(
    document_id="project_alpha",
    chunks=chunks
)

print("Stored successfully")