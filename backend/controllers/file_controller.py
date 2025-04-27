from flask import Blueprint, request, jsonify
from utils.file_handler import handle_file_upload
from utils.text_processor import process_text
from dtos.response_api import format_response

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
        
        return format_response(processed_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
