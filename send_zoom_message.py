import os
import requests

def send_chat_to_zoom(message, payload):
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

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Successfully sent chat to Zoom.")
    except Exception as e:
        print(f"Error sending chat to Zoom: {e}")

def get_chatbot_token():
    try:
        url = "https://zoom.us/oauth/token?grant_type=client_credentials"
        client_id = os.getenv("ZOOM_CLIENT_ID")
        client_secret = os.getenv("ZOOM_CLIENT_SECRET")

        auth = (client_id, client_secret)
        response = requests.post(url, auth=auth)
        response.raise_for_status()

        return response.json()['access_token']
    except Exception as e:
        print(f"Error getting chatbot token: {e}")
        raise
