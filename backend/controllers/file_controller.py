from flask import Blueprint, request, jsonify
from utils.file_handler import handle_file_upload
from utils.text_processor import process_text
from dtos.response_api import response_api
from utils.shared_resources import faiss_db

file_controller = Blueprint('file_controller', __name__)

@file_controller.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        text = handle_file_upload(file)
        processed_text = process_text(text)
        chunks = chunk_text(processed_text)

        for idx, chunk in enumerate(chunks):
            metadata = {
                "filename": file.filename,
                "chunk_id": idx
            }
            faiss_db.store_embeddings(chunk, metadata)
        
        return response_api(f"Uploaded {len(chunks)} chunks from {file.filename} successfully.")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def chunk_text(text, chunk_size=500):
    """
    Split text into chunks of given size (in words).
    """
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
