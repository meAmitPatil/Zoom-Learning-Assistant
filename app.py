import os
import base64
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Flask app, Qdrant client, and Sentence Transformer model
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="localhost", port=6333)

# Fireworks AI configuration
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
fireworks_model_endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Store user quiz answers
user_quiz_answers = {}

# Function to generate embeddings
def embed_text(text):
    return model.encode(text)

# Function to query Qdrant for meeting content
def query_qdrant(query_text):
    with open("transcripts/transcript_intro_to_ml.txt", "r") as file:
        meeting_text = file.read()
    return meeting_text

# Function to generate a summary from the lecture transcript
def generate_summary(text):
    max_chunk = 512
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    
    summary = ""
    for chunk in text_chunks:
        summary_chunk = summarizer(chunk, max_length=min(150, max(30, int(len(chunk) * 0.5))), min_length=30, do_sample=False)
        summary += summary_chunk[0]['summary_text'] + " "
    
    return summary.strip()

# Function to generate a quiz question using Fireworks API
def generate_quiz_with_fireworks(summary_text):
    headers = {
        "Authorization": f"Bearer {fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",  # Ensure this is the correct model path
        "messages": [
            {
                "role": "user",
                "content": f"Generate a multiple-choice quiz question based on the following content:\n\n{summary_text}"
            }
        ]
    }
    
    response = requests.post(fireworks_model_endpoint, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        question_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Parse the response to extract the question and the correct answer
        correct_answer = None
        if "Correct answer:" in question_text:
            question_text, correct_answer = question_text.split("**Correct answer:**")[0], question_text.split("**Correct answer:**")[-1].strip()
            correct_answer = correct_answer.split(")")[0][-1]  # Extract only the letter (A, B, C, etc.)

        return question_text.strip(), correct_answer
    else:
        print("Error calling Fireworks API:", response.status_code, response.text)
        return "Unable to generate a question at this time.", None

# Function to get Zoom bot token
def get_zoom_bot_token():
    client_id = os.getenv("ZOOM_CLIENT_ID")
    client_secret = os.getenv("ZOOM_CLIENT_SECRET")
    token_url = "https://zoom.us/oauth/token?grant_type=client_credentials"
    
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    }
    
    response = requests.post(token_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print("Error getting Zoom bot token:", response.json())
        return None

# Handle Zoom webhook commands
def handle_zoom_webhook(payload):
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()

    command_text = payload.get("payload", {}).get("cmd")
    if not command_text:
        print("No 'cmd' key found in the payload.")
        return

    # Check for the /summarize command
    if "/summarize lecture" in command_text.lower():
        transcript = query_qdrant("today's lecture")
        summary = generate_summary(transcript)
        response_text = f"Lecture Summary:\n{summary}"

    # Check for the /start quiz command
    elif "/start quiz" in command_text.lower():
        transcript = query_qdrant("today's lecture")
        summary = generate_summary(transcript)
        question_text, correct_answer = generate_quiz_with_fireworks(summary)
        
        # Store the correct answer dynamically for validation in the user's session or memory
        global current_correct_answer
        current_correct_answer = correct_answer
        
        response_text = f"Quiz Question:\n{question_text}"

    # Check for answer submission command by users
    elif "answer" in command_text.lower():
        # Extract the answer from the command
        answer_given = command_text.lower().split("answer")[-1].strip()
        
        # Validate the user's answer with the stored correct answer
        if answer_given.lower() == current_correct_answer.lower():
            response_text = "Correct answer!"
        else:
            response_text = "Incorrect answer. Try again."

    else:
        response_text = "Command not recognized. Try '/summarize lecture' or '/start quiz'."

    # Send response to Zoom
    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": payload["payload"]["toJid"],
        "user_jid": payload["payload"]["toJid"],
        "account_id": payload["payload"]["accountId"],
        "content": {
            "head": {"text": "Meeting Insights - Response"},
            "body": [{"type": "message", "text": response_text}]
        }
    }

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    zoom_url = "https://api.zoom.us/v2/im/chat/messages"
    resp = requests.post(zoom_url, headers=headers, json=message_payload)

    if resp.status_code != 201:
        print("Error sending message to Zoom:", resp.json())
    else:
        print("Message sent successfully to Zoom:", resp.json())

# Flask route to handle Zoom webhook POST requests
@app.route('/webhook', methods=['POST'])
def zoom_webhook():
    payload = request.json
    handle_zoom_webhook(payload)
    return jsonify({"status": "Processed"}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 4000)))
