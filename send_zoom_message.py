import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def send_chat_to_zoom(message, payload):
    """
    Sends a chat message to Zoom via the Zoom chat API.
    """
    try:
        url = "https://api.zoom.us/v2/im/chat/messages"
        token = get_chatbot_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        data = {
            "robot_jid": os.getenv("ZOOM_BOT_JID"),
            "to_jid": payload['toJid'],
            "user_jid": payload['toJid'],
            "content": {
                "head": {"text": "OpenAI Chatbot"},
                "body": [{"type": "message", "text": f"Assistant: {message}"}]
            }
        }

        # Send the request
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Successfully sent chat to Zoom.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending chat to Zoom: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def get_chatbot_token():
    """
    Retrieves the chatbot token using client credentials from the Zoom OAuth API.
    """
    try:
        url = "https://zoom.us/oauth/token?grant_type=client_credentials"
        client_id = os.getenv("ZOOM_CLIENT_ID")
        client_secret = os.getenv("ZOOM_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError("Missing Zoom client ID or client secret in environment variables.")

        headers = {
            "Authorization": "Basic " + requests.auth._basic_auth_str(client_id, client_secret)
        }

        response = requests.post(url, headers=headers)
        response.raise_for_status()

        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"Error getting chatbot token from Zoom: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error when retrieving chatbot token: {e}")
        raise
