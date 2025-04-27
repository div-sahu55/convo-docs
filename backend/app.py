from flask import Flask
from controllers.query_controller import query_bp

app = Flask(__name__)

# Register Blueprints
app.register_blueprint(query_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
