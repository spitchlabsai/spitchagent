from typing import List, Dict, Any
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer


class RAGService:
    def __init__(self, supabase_url: str, supabase_key: str, embedder=None):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        # self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model = embedder

    # -------------------------------
    # Document Upload + Chunking
    # -------------------------------
    def embed_and_store_document(self, user_id: str, filename: str, text: str, chunk_size: int = 500):
        """Splits text into chunks, embeds, and stores in DB"""
        # Create document entry
        doc = (
            self.supabase.table("documents")
            .insert({"user_id": user_id, "filename": filename})
            .execute()
        ).data[0]

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = self.model.encode(chunks, convert_to_numpy=True).tolist()

        # Store chunks
        rows = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append(
                {
                    "document_id": doc["id"],
                    "user_id": user_id,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "embedding": emb,
                }
            )

        self.supabase.table("document_chunks").insert(rows).execute()

        return {"document_id": doc["id"], "chunks": len(rows)}

    # -------------------------------
    # Semantic Search
    # -------------------------------
    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search relevant chunks for a user"""
        query_embedding = self.model.encode(query).tolist()

        response = (
            self.supabase.rpc(
                "search_document_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "user_id_param": user_id,
                },
            )
            .execute()
        )

        return response.data or []

    # -------------------------------
    # Context Builder
    # -------------------------------
    def get_context(self, query: str, user_id: str, max_chunks: int = 5) -> str:
        """Return combined context string for a query"""
        chunks = self.search(query, user_id, max_chunks)

        if not chunks:
            return ""

        parts = []
        for c in chunks:
            parts.append(f"[Chunk]\n{c['chunk_text']} (score={round(c['similarity'],3)})")

        return "\n\n---\n\n".join(parts)
