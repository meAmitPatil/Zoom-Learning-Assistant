import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from zoom_webhook_handler import handle_zoom_webhook

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "OK", 200

@app.route('/webhook', methods=['POST'])
def zoom_webhook():
    return handle_zoom_webhook(request)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 4000))
    app.run(host='0.0.0.0', port=port)
