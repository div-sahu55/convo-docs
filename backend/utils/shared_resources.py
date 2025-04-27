from utils.vector_store import FAISSDB

faiss_db = FAISSDB(db_path="./faiss_db_data", collection_name="document_embeddings")