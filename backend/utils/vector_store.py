import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import hashlib

class FAISSDB:
    """
    A simple in-memory vector database using FAISS and Sentence Transformers.
    Does not currently persist the FAISS index or data to disk.
    """
    def __init__(self, db_path="./db", collection_name="mistral_embeddings"):
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize sentence-transformer model
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()

        # Setup FAISS index (flat index using L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)

        # Map from index to IDs and metadata
        # These lists and dictionary store the original data corresponding
        # to the embeddings stored in the FAISS index.
        # The index in these lists/dict corresponds directly to the index in FAISS.
        self.ids = []  # list of IDs (in order of insertion)
        self.documents = []  # list of documents (in order of insertion)
        self.metadata_store = {}  # id -> metadata mapping

        print(f"FAISSDB initialized. Using collection '{collection_name}'.")
        print(f"Embedding dimension: {self.embedding_dimension}")

    def _generate_id_from_text(self, text: str) -> str:
        """
        Generates a unique ID for a document based on its text content
        using SHA256 hashing.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate and return numpy array embedding for a given text.
        Ensures the embedding is float32, which is required by FAISS.
        """
        # encode returns a numpy array of shape (batch_size, embedding_dimension)
        embedding_np = self.embedder.encode([text])[0] # Get the first (and only) embedding
        return embedding_np.astype(np.float32) # make sure dtype is float32

    def store_embeddings(self, text: str, metadata: Dict):
        """
        Generates an embedding for the text, creates an ID, and adds
        the embedding, text, ID, and metadata to the database.

        Note: This simple implementation does not prevent adding duplicate texts.
        Each addition will create a new entry in the FAISS index and associated lists.
        """
        embedding = self.generate_embeddings(text)
        embedding_np = embedding.reshape(1, -1)  # Reshape to (1, dimension) for FAISS add

        doc_id = self._generate_id_from_text(text)

        # Add the embedding to the FAISS index
        # The index where it's added corresponds to len(self.ids) before appending
        self.index.add(embedding_np)

        # Store the corresponding data
        self.ids.append(doc_id)
        self.documents.append(text)
        # Store metadata, ensuring values are strings (optional but good practice)
        self.metadata_store[doc_id] = {k: str(v) for k, v in metadata.items()}


        print(f"Stored document. ID: {doc_id[:8]}...") # Print truncated ID

    def query_embeddings(self, query: str, file_name : str) -> List[Dict]:
        """
        Queries the database for documents similar to the given query text.

        Args:
            query: The text to query for.
            n_results: The maximum number of results to return.

        Returns:
            A list of dictionaries, where each dictionary contains details
            ('id', 'distance', 'document', 'metadata') of a similar document,
            sorted by similarity (lowest distance first).
        """
        n_results = 5;
        if self.index.ntotal == 0:
            print("Database is empty. No results.")
            return []

        query_embedding = self.generate_embeddings(query).reshape(1, -1)

        # Perform the search
        # distances: numpy array of shape (num_queries, n_results)
        # indices: numpy array of shape (num_queries, n_results)
        distances, indices = self.index.search(query_embedding, n_results)

        valid_results = []

        # Iterate through the results for the first (and only) query
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: # FAISS returns -1 for results beyond the total number of vectors
                continue

            # Retrieve the stored data using the index returned by FAISS
            try:
                doc_id = self.ids[idx]
                document = self.documents[idx]
                metadata = self.metadata_store.get(doc_id, {}) # Use .get for safety
                result = {
                    'id': doc_id,
                    'distance': dist,
                    'document': document,
                    'metadata': metadata
                }
                valid_results.append(result)
            except IndexError:
                 # This case should ideally not happen if indices are valid,
                 # but included for robustness.
                 print(f"Warning: FAISS returned an invalid index {idx}.")
                 continue


        # Note: Results are already sorted by distance by FAISS search
        return valid_results

    # --- Persistence Methods (Optional - requires saving/loading index and data) ---
    # Adding placeholders to show where this functionality would go
    def save_index(self):
         """
         Saves the FAISS index to a file.
         Requires saving the index itself and the associated metadata/data separately.
         """
         # Example: requires saving self.index, self.ids, self.documents, self.metadata_store
         # FAISS index can be saved using faiss.write_index(self.index, index_file_path)
         # Data (ids, documents, metadata_store) can be saved using pickle or json
         print("Save functionality not implemented in this example.")
         pass # Implementation needed

    def load_index(self):
        """
        Loads the FAISS index and associated data from files.
        """
        # Example: requires loading self.index, self.ids, self.documents, self.metadata_store
        # FAISS index can be loaded using faiss.read_index(index_file_path)
        # Data needs to be loaded from the format they were saved in
        print("Load functionality not implemented in this example.")
        # After loading, ensure self.embedding_dimension matches the loaded index dimension
        # self.embedding_dimension = self.index.d
        pass # Implementation needed


# Example Usage
if __name__ == '__main__':
    # Clean up previous runs if necessary (optional)
    # db_dir = "./faiss_db"
    # if os.path.exists(db_dir):
    #     # Add code to clean up if saving files were implemented
    #     pass


    db = FAISSDB(db_path="./faiss_db", collection_name="document_embeddings")

    docs_to_store = [
        {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "example1", "topic": "animals"}},
        {"text": "Exploring the capabilities of vector databases.", "metadata": {"source": "tech_blog", "topic": "database"}},
        {"text": "Sentence transformers provide useful text embeddings.", "metadata": {"source": "ml_paper", "topic": "nlp"}},
        {"text": "A lazy dog sat under the tree.", "metadata": {"source": "example2", "topic": "animals"}},
        {"text": "Artificial intelligence is transforming industries.", "metadata": {"source": "news", "topic": "technology"}}
    ]

    print("\n--- Storing Documents ---")
    for doc in docs_to_store:
        # Metadata conversion to string values is handled inside store_embeddings
        db.store_embeddings(text=doc['text'], metadata=doc['metadata'])

    print(f"\nTotal documents in DB: {db.index.ntotal}")


    print("\n--- Querying Documents ---")
    query_text = "Tell me about inactive animals."
    # Correctly passing n_results
    similar_docs = db.query_embeddings(query=query_text, n_results=3)

    if similar_docs:
        print(f"\nFound {len(similar_docs)} similar documents for query: '{query_text}'")
        for i, doc in enumerate(similar_docs):
            print(f"\n  Result {i+1}:")
            print(f"    ID: {doc['id'][:8]}...") # Print truncated ID
            print(f"    Distance: {doc['distance']:.4f}")
            print(f"    Document: {doc['document']}")
            print(f"    Metadata: {doc['metadata']}")
    else:
        print(f"No matching documents found for query: '{query_text}'")

    print("\n--- Querying with a different query ---")
    query_text_2 = "What is a vector database?"
    similar_docs_2 = db.query_embeddings(query=query_text_2, n_results=2)

    if similar_docs_2:
        print(f"\nFound {len(similar_docs_2)} similar documents for query: '{query_text_2}'")
        for i, doc in enumerate(similar_docs_2):
            print(f"\n  Result {i+1}:")
            print(f"    ID: {doc['id'][:8]}...") # Print truncated ID
            print(f"    Distance: {doc['distance']:.4f}")
            print(f"    Document: {doc['document']}")
            print(f"    Metadata: {doc['metadata']}")
    else:
        print(f"No matching documents found for query: '{query_text_2}'")

    # Example demonstrating no results if DB is empty (conceptually - need to re-init)
    # print("\n--- Demonstrating empty DB query ---")
    # empty_db = FAISSDB()
    # empty_results = empty_db.query_embeddings("test query")
    # print(f"Results from empty DB: {empty_results}")