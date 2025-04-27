from flask import jsonify

def response_api(data):
    """Format the response to be returned to the frontend."""
    response = {
        "status": "success",
        "data": data
    }
    return jsonify(response)
