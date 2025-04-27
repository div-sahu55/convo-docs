from flask import Blueprint, request, jsonify
import traceback
from models.llama_model import LlamaModel
from utils.shared_resources import faiss_db

query_bp = Blueprint('query_bp', __name__)

model = LlamaModel()

@query_bp.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        filename = data.get('filename')

        if not prompt or not filename:
            return jsonify({"error": "Prompt and filename are required"}), 400
        
        retrieved_chunk = faiss_db.query_embeddings(prompt, filename)
        print("Retrieved Chunk:", retrieved_chunk)

        unique_documents = {chunk['document'] for chunk in retrieved_chunk}
        clean_retrieved_chunk = " ".join(unique_documents)

        final_prompt = (
            f"[INST] Use the following context to answer the user's query.\n\n"
            f"Context:\n{clean_retrieved_chunk}\n\n"
            f"Query:\n{prompt}\n\n"
            f"Analyze if the query is related to the context or not"
            f" if it is, then answer in a helpful and concise manner. Else, decline the request politely. [/INST]"
        )

        print("Final Prompt:", final_prompt)

        response = model.query(final_prompt)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
