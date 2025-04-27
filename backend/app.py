from flask import Flask
from controllers.query_controller import query_bp
from controllers.file_controller import file_controller
from utils.shared_resources import faiss_db

app = Flask(__name__)


# Register Blueprints
app.register_blueprint(query_bp, url_prefix='/api')
app.register_blueprint(file_controller, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
