import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class ChromaDB:
    def __init__(self, db_path="db"):
        # Initialize Chroma client and collection
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_path))
        self.collection = self.client.create_collection("mistral_embeddings")

        # Initialize the sentence-transformer model for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a given text using the Sentence-Transformer model.
        """
        return self.embedder.encode([text])[0]

    def store_embeddings(self, text: str, metadata: dict):
        """
        Store the embeddings of the text in the ChromaDB.
        """
        embedding = self.generate_embeddings(text)
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding]
        )

    def query_embeddings(self, query: str) -> str:
        """
        Query ChromaDB for similar responses.
        """
        query_embedding = self.generate_embeddings(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        if results['documents']:
            return results['documents'][0]
        else:
            return "No matching documents found."
