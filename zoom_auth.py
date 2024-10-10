import os
import base64
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_zoom_bot_token():
    """
    Retrieves the Zoom bot token using client credentials from the Zoom OAuth API.
    """
    client_id = os.getenv("ZOOM_CLIENT_ID")
    client_secret = os.getenv("ZOOM_CLIENT_SECRET")
    
    # Check if client_id and client_secret are present
    if not client_id or not client_secret:
        print("Error: Missing Zoom client ID or client secret in environment variables.")
        return None

    token_url = "https://zoom.us/oauth/token?grant_type=client_credentials"
    
    # Prepare the authorization header
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}"
    }

    try:
        response = requests.post(token_url, headers=headers)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Error getting Zoom bot token: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
