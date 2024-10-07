import os
import json
import requests
import spacy
from dateparser.search import search_dates
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import base64

# Load environment variables
load_dotenv()

# Initialize Flask app, spaCy model, and Qdrant client
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(":memory:")

# Function to generate embeddings
def embed_text(text):
    return model.encode(text)

# Function to extract action items, names, and deadlines from meeting text
def extract_action_items(meeting_text):
    action_items = []

    # Process the text with spaCy
    doc = nlp(meeting_text)
    sentences = list(doc.sents)

    for sentence in sentences:
        # Extract verbs (actions) and names dynamically
        verbs = [token.lemma_ for token in sentence if token.pos_ == "VERB"]
        names = [ent.text for ent in sentence.ents if ent.label_ == "PERSON"]

        # Parse for dates and deadlines
        dates = search_dates(sentence.text)
        deadline = dates[0][1].strftime("%Y-%m-%d") if dates else "No specific deadline"

        # If we found verbs, assume it's an action item
        if verbs:
            action_text = sentence.text.strip()
            responsible_person = ", ".join(names) if names else "Unassigned"
            item = f"- **{action_text}** - **Assigned to: {responsible_person}** - **Deadline: {deadline}**"
            action_items.append(item)

    return action_items

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
        print(f"Stored point {i} with title '{title}' in Qdrant")

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
def query_qdrant(command_text):
    query_vector = embed_text(command_text)

    try:
        search_results = qdrant_client.search(
            collection_name="meeting_transcripts",
            query_vector=query_vector,
            limit=1
        )

        if search_results:
            return search_results[0].payload["content"]
        else:
            return "No relevant meeting information found."
    except Exception as e:
        print("Error during Qdrant query:", e)
        return "An error occurred while processing your request."

# Function to handle the Zoom webhook
def handle_zoom_webhook(payload):
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()

    # Check for the 'cmd' key within the nested 'payload'
    command_text = payload.get("payload", {}).get("cmd")
    if not command_text:
        print("No 'cmd' key found in the payload. Payload received:", payload)
        return  # Exit the function if 'cmd' is missing

    # Retrieve the relevant meeting transcript
    meeting_text = query_qdrant(command_text)

    # Extract action items, responsibilities, and deadlines
    action_items = extract_action_items(meeting_text)
    action_list_text = "\n".join(action_items) if action_items else "No action items found."

    # Format the bot's response with the action list
    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": payload["payload"]["toJid"],
        "user_jid": payload["payload"]["toJid"],
        "account_id": payload["payload"]["accountId"],
        "content": {
            "head": {
                "text": "Meeting Insights - Action Items"
            },
            "body": [
                {
                    "type": "message",
                    "text": action_list_text
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
