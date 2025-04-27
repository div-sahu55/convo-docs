from flask import Blueprint, request, jsonify
from models.llama_model import GPT2Model

query_bp = Blueprint('query_bp', __name__)

# Initialize the model
model = GPT2Model()

@query_bp.route('/query', methods=['POST'])
def query():
    try:
        # Get user prompt from the request
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Get model response
        response = model.query(prompt)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
