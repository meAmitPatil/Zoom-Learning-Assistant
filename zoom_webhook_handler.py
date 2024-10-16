import os
import requests
import base64
from flask import jsonify
from dotenv import load_dotenv

load_dotenv()

def handle_zoom_webhook(response, payload):
    """
    Handles incoming requests from the Zoom webhook, generating a response
    to be sent back to the Zoom chat.

    :param response: Text response to be sent back to the Zoom chat
    :param payload: JSON payload from the Zoom webhook, containing details like 'toJid' and 'accountId'
    """
    zoom_bot_jid = os.getenv("ZOOM_BOT_JID")
    auth_token = get_zoom_bot_token()

    if not auth_token:
        print("Error: Could not retrieve Zoom bot token.")
        return

    if not zoom_bot_jid:
        print("Error: ZOOM_BOT_JID environment variable not found.")
        return

    message_payload = {
        "robot_jid": zoom_bot_jid,
        "to_jid": payload.get("toJid"),
        "user_jid": payload.get("toJid"),
        "account_id": payload.get("accountId"),
        "content": {
            "head": {"text": "Meeting Insights"},
            "body": [
                {"type": "message", "text": response}
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    zoom_url = "https://api.zoom.us/v2/im/chat/messages"

    try:
        resp = requests.post(zoom_url, headers=headers, json=message_payload)
        resp.raise_for_status()
        print("Message sent successfully to Zoom:", resp.json())
    except requests.exceptions.RequestException as e:
        print(f"Error sending message to Zoom: {e}")
        if resp.content:
            print("Response content:", resp.json())

def get_zoom_bot_token():
    """
    Retrieves the Zoom bot token using client credentials.
    """
    zoom_client_id = os.getenv("ZOOM_CLIENT_ID")
    zoom_client_secret = os.getenv("ZOOM_CLIENT_SECRET")

    if not zoom_client_id or not zoom_client_secret:
        print("Error: Missing ZOOM_CLIENT_ID or ZOOM_CLIENT_SECRET environment variables.")
        return None

    auth_url = "https://zoom.us/oauth/token?grant_type=client_credentials"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{zoom_client_id}:{zoom_client_secret}".encode()).decode()
    }

    try:
        response = requests.post(auth_url, headers=headers)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Zoom bot token: {e}")
        if response.content:
            print("Response content:", response.json())
        return None
