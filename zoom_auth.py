# zoom_auth.py
import os
import base64
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
