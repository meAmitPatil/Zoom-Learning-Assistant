import os
import json
import requests
import spacy
import base64
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Flask app, spaCy model, Qdrant client, and Sentence Transformer model
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(":memory:")

# Load the open-source LLM for question-answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to generate embeddings
def embed_text(text):
    return model.encode(text)

# Function to store meeting data in Qdrant
def store_meeting_data():
    with open('meeting_data.json', 'r') as file:
        meeting_data = json.load(file)

    if qdrant_client.collection_exists("meeting_transcripts"):
        qdrant_client.delete_collection(collection_name="meeting_transcripts")

    qdrant_client.create_collection(
        collection_name="meeting_transcripts",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    for i, (title, content) in enumerate(meeting_data.items()):
        embedding = embed_text(content)
        point = PointStruct(id=i, vector=embedding, payload={"title": title, "content": content})
        qdrant_client.upsert(collection_name="meeting_transcripts", points=[point])
        print(f"Stored point {i} with title '{title}' and content: {content[:100]}...")  # Log the first 100 chars of each content

# Function to get the Zoom bot token
def get_zoom_bot_token():
    client_id = os.getenv("ZOOM_CLIENT_ID")
    client_secret = os.getenv("ZOOM_CLIENT_SECRET")
    token_url = "https://zoom.us/oauth/token?grant_type=client_credentials"

    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    }

    response = requests.post(token_url, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print("Error getting Zoom bot token:", response.json())
        return None

# Function to query Qdrant for meeting content
def query_qdrant(query_text):
    search_result = qdrant_client.search(
        collection_name="meeting_transcripts",
        query_vector=embed_text(query_text),
        limit=10
    )
    print("Search Results:", search_result)  # Log search results for debugging
    meeting_text = " ".join(result.payload["content"] for result in search_result if "content" in result.payload)
    return meeting_text if meeting_text else "No relevant information found."

# Function to filter sentences based on keywords in the question
def answer_question(meeting_text, question):
    keywords = question.lower().split()
    
    # Filter sentences based on question keywords
    doc = nlp(meeting_text)
    sentences = [sent.text for sent in doc.sents]
    relevant_sentences = [
        sent for sent in sentences if any(keyword in sent.lower() for keyword in keywords)
    ]
    
    # Join relevant sentences for a refined context
    refined_context = " ".join(relevant_sentences)
    qa_input = {
        "question": question,
        "context": refined_context
    }
    response = qa_pipeline(qa_input)
    return response.get("answer", "No relevant information found.")

# Function to handle the Zoom webhook
def handle_zoom_webhook(payload):
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()

    command_text = payload.get("payload", {}).get("cmd")
    if not command_text:
        print("No 'cmd' key found in the payload. Payload received:", payload)
        return

    if "sales" in command_text.lower():
        query_topic = "sales projection"
    elif "product launch" in command_text.lower():
        query_topic = "product launch"
    else:
        query_topic = "general"

    print(f"Querying Qdrant for topic: {query_topic} with command: {command_text}")  # Log topic and command
    meeting_text = query_qdrant(query_topic)
    answer = answer_question(meeting_text, command_text)

    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": payload["payload"]["toJid"],
        "user_jid": payload["payload"]["toJid"],
        "account_id": payload["payload"]["accountId"],
        "content": {
            "head": {
                "text": f"Meeting Insights - {query_topic.capitalize()}"
            },
            "body": [
                {
                    "type": "message",
                    "text": f"Question: {command_text}\nAnswer: {answer}"
                }
            ]
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
    store_meeting_data()  # Load data into Qdrant
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 4000)))
