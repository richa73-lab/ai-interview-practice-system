# rag/embedder.py

from sentence_transformers import SentenceTransformer


# load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks


def create_embeddings(chunks):
    """
    Convert text chunks into embeddings (vectors)
    """

    embeddings = model.encode(chunks)

    return embeddings
