import os
import re
import base64
import requests
import spacy
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from chatgroq_handler import call_groq_api
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.settings import Settings
from google_drive_upload import save_notes_to_drive

Settings.llm = None

load_dotenv()

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="localhost", port=6333)
huggingface_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
fireworks_model_endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
user_quiz_answers = {}
current_correct_answer = []
poll_rating = {}

def embed_text(text):
    return model.encode(text)

def store_transcript(file_path):
    try:
        with open(file_path, "r") as file:
            transcript_text = file.read()
        max_chunk_size = 512
        transcript_chunks = [transcript_text[i:i + max_chunk_size] for i in range(0, len(transcript_text), max_chunk_size)]
        embeddings = [embed_text(chunk) for chunk in transcript_chunks]
        if qdrant_client.collection_exists(collection_name="lecture_transcripts"):
            qdrant_client.delete_collection(collection_name="lecture_transcripts")
        qdrant_client.create_collection(
            collection_name="lecture_transcripts",
            vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine")
        )
        points = [PointStruct(id=i, vector=embedding, payload={"text": chunk}) for i, (chunk, embedding) in enumerate(zip(transcript_chunks, embeddings))]
        qdrant_client.upsert(collection_name="lecture_transcripts", points=points)
        print("Transcript stored in Qdrant successfully.")
        
        documents = [Document(text=chunk) for chunk in transcript_chunks]
        global llama_index
        llama_index = VectorStoreIndex.from_documents(documents, embed_model=huggingface_embed_model)
        print("Transcript stored in LlamaIndex successfully.")
    except Exception as e:
        print("An error occurred while storing transcript:", e)

def query_qdrant_for_summary(query_text):
    query_vector = embed_text(query_text)
    search_result = qdrant_client.search(
        collection_name="lecture_transcripts",
        query_vector=query_vector,
        limit=5
    )
    combined_text = " ".join([hit.payload["text"] for hit in search_result])
    return combined_text

def query_llamaindex_for_summary(query_text):
    query_engine = llama_index.as_query_engine(llm=None)
    response = query_engine.query(query_text)
    combined_text = response.response
    return combined_text

def generate_summary_with_llamaindex(query_text):
    relevant_text = query_llamaindex_for_summary(query_text)
    raw_summary = generate_summary(relevant_text)
    final_summary = complete_summary(raw_summary)
    return final_summary

def generate_summary(text):
    max_chunk = 512
    text_chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in text_chunks:
        chunk_max_length = min(150, int(len(chunk) * 0.75))
        summary_chunk = summarizer(chunk, max_length=chunk_max_length, min_length=30, do_sample=False)
        summary += summary_chunk[0]['summary_text'] + " "
    return complete_summary(summary.strip())

def complete_summary(summary):
    if not re.search(r'[.!?]$', summary.strip()):
        summary = summary.rsplit(' ', 1)[0] + '.'
    return summary

def generate_quiz_with_fireworks(summary_text):
    headers = {
        "Authorization": f"Bearer {fireworks_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
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
        if "Correct answer:" in question_text:
            question, correct_answer = question_text.split("Correct answer:")
            question = question.strip()
            correct_answer = correct_answer.strip().split(")")[0]
            current_correct_answer.clear()
            current_correct_answer.append(correct_answer)
            return question
        else:
            return "Unable to generate a question at this time."
    else:
        return "Unable to generate a question at this time."
    
def generate_notes_with_fireworks(transcript_file):
    try:
        with open(transcript_file, "r") as file:
            transcript_text = file.read()
    except FileNotFoundError:
        print(f"Error: The file {transcript_file} was not found.")
        return None
    data = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Create detailed lecture notes based on the following transcript. "
                    "Include important points, key concepts, and any relevant details that will help with studying:\n\n" +
                    transcript_text
                )
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {fireworks_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(fireworks_model_endpoint, json=data, headers=headers)
        response.raise_for_status()
        notes_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return notes_content
    except requests.exceptions.RequestException as e:
        print(f"Error generating notes with Fireworks AI: {e}")
        return None

def get_topic_and_resources_with_llm(summary_text):
    headers = {
        "Authorization": f"Bearer {fireworks_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "accounts/fireworks/models/firefunction-v2",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Based on the following lecture content, identify the main topic "
                    f"and provide a list of relevant educational resources such as articles, "
                    f"videos, or books to help a student learn more about it. "
                    f"Please follow this exact format:\n\n"
                    f"Main Topic: [Topic Name]\n"
                    f"Resources:\n"
                    f"1) [Resource Title] - [Description]\n"
                    f"2) [Resource Title] - [Description]\n\n"
                    f"Lecture Content:\n{summary_text}"
                )
            }
        ]
    }
    response = requests.post(fireworks_model_endpoint, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        lines = response_text.splitlines()
        topic = ""
        resources = []
        for line in lines:
            line = line.strip()
            if line.startswith("Main Topic:"):
                topic = line.replace("Main Topic:", "").strip()
            elif line.startswith(("1)", "2)", "3)", "4)", "5)")):
                resource_text = line.split(")", 1)[-1].strip()
                if resource_text:
                    resources.append(resource_text)
        if not topic:
            topic = "Unknown Topic"
        if not resources:
            resources = ["No resources found for this topic."]
        return topic, resources
    else:
        print("Error with Fireworks API:", response.status_code, response.text)
        return None, None

def get_zoom_bot_token():
    client_id = os.getenv("ZOOM_CLIENT_ID")
    client_secret = os.getenv("ZOOM_CLIENT_SECRET")
    token_url = "https://zoom.us/oauth/token?grant_type=client_credentials"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    }
    response = requests.post(token_url, headers=headers)
    return response.json().get("access_token") if response.status_code == 200 else None

def handle_zoom_webhook(payload):
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()
    payload_data = payload.get("payload", {})
    command_text = payload_data.get("cmd", "").lower()
    to_jid = payload_data.get("toJid")
    account_id = payload_data.get("accountId")
    if not to_jid or not account_id:
        print("Error: 'toJid' or 'accountId' is missing in the payload")
        return "Error: Missing required fields in the payload"
    response_text = "Command not recognized. Try '/summarize lecture' or '/start quiz'."
    if "/summarize lecture" in command_text:
        summary = generate_summary_with_llamaindex("lecture summary")
        response_text = f"Lecture Summary:\n{summary}"
    elif "/start quiz" in command_text:
        summary = generate_summary_with_llamaindex("lecture summary")
        question_text = generate_quiz_with_fireworks(summary)
        response_text = f"Quiz Question:\n{question_text}"
    elif "/answer" in command_text:
        answer_given = command_text.split("/answer")[-1].strip().upper()
        correct_answer = current_correct_answer[-1].upper()
        response_text = "Correct answer!" if answer_given == correct_answer else "Incorrect answer. Try again."
    elif "/additional resources" in command_text:
        summary = generate_summary_with_llamaindex("lecture summary")
        topic, resources = get_topic_and_resources_with_llm(summary)
        if resources:
            response_text = f"**Main Topic:** {topic}\n\n**Additional Resources:**\n" + "\n".join([f"{i+1}. {res}" for i, res in enumerate(resources)])
        else:
            response_text = "Sorry, no additional resources found for this topic."
    elif "/poll" in command_text:
        response_text = "On a scale of 1-5, how well did you understand the lecture? Please respond with '/rate <number>'."
    elif command_text.startswith("/rate"):
        try:
            rating = int(command_text.split("/rate")[-1].strip())
            if 1 <= rating <= 5:
                poll_rating[to_jid] = rating
                response_text = f"Thank you! You rated the lecture with a {rating} out of 5."
            else:
                response_text = "Please enter a valid rating between 1 and 5."
        except ValueError:
            response_text = "Invalid rating. Please respond with a number between 1 and 5."
    elif "/generate notes" in command_text:
        transcript_file = "transcripts/transcript_intro_to_ml.txt"
        notes = generate_notes_with_fireworks(transcript_file)
        if notes:
            save_notes_to_drive(notes, filename="Lecture_Notes.txt", folder_id = "1lcClr2x2N2v8iQcSPEXrlgVnXGB6Ym7d")
            response_text = "Notes generated successfully, sent to Zoom chat, and saved to Google Drive."
        else:
            response_text = "Failed to generate notes from the transcript."
    else:
        response_text = call_groq_api(command_text, to_jid)
    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": to_jid,
        "user_jid": to_jid,
        "account_id": account_id,
        "content": {
            "head": {"text": "Learning Assistant - Response"},
            "body": [{"type": "message", "text": response_text}]
        }
    }
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    zoom_url = "https://api.zoom.us/v2/im/chat/messages"
    response = requests.post(zoom_url, headers=headers, json=message_payload)
    print("Message sent successfully to Zoom:", response.json() if response.status_code == 201 else response.json())

@app.route('/webhook', methods=['POST'])
def zoom_webhook():
    payload = request.json
    handle_zoom_webhook(payload)
    return jsonify({"status": "Processed"}), 200

if __name__ == '__main__':
    store_transcript("transcripts/transcript_intro_to_ml.txt")
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 4000)))
