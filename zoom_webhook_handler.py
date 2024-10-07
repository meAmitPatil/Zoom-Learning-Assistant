import os
import requests
import base64
from flask import jsonify

def handle_zoom_webhook(response, payload):
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()

    # Define the payload for sending a message to Zoom chat
    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": payload["toJid"],
        "user_jid": payload["toJid"],  # Adding the user_jid field
        "account_id": payload["accountId"],
        "content": {
            "head": {
                "text": "Meeting Insights"
            },
            "body": [
                {
                    "type": "message",
                    "text": response  # Inject the response from Qdrant here
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

def get_zoom_bot_token():
    zoom_client_id = os.getenv("ZOOM_CLIENT_ID")
    zoom_client_secret = os.getenv("ZOOM_CLIENT_SECRET")

    auth_url = "https://zoom.us/oauth/token?grant_type=client_credentials"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{zoom_client_id}:{zoom_client_secret}".encode()).decode()
    }

    response = requests.post(auth_url, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print("Error retrieving Zoom bot token:", response.json())
        return None
