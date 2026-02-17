import faiss
import numpy as np


class VectorStore:
    def __init__(self, embeddings, chunks):
        """
        embeddings: numpy array of vectors
        chunks: original text chunks
        """

        self.chunks = chunks

        # get embedding dimension (384)
        dimension = embeddings.shape[1]

        # create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # add embeddings to index
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, top_k=3):
        """
        Returns most relevant chunks for a query
        """

        distances, indices = self.index.search(
            np.array([query_embedding]), top_k
        )

        results = [self.chunks[i] for i in indices[0]]

        return results
