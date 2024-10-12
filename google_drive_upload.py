from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def save_notes_to_drive(notes_text, filename="Generated_Notes.txt", folder_id=None):
    creds = service_account.Credentials.from_service_account_file('client_secrets.json')
    service = build('drive', 'v3', credentials=creds)

    # Save notes as a temporary file
    with open(filename, 'w') as f:
        f.write(notes_text)

    # Define file metadata and include the folder ID if provided
    file_metadata = {
        'name': filename,
        'mimeType': 'text/plain'
    }
    
    if folder_id:
        file_metadata['parents'] = [folder_id]  # Set the parent folder

    media = MediaFileUpload(filename, mimetype='text/plain')

    # Upload the file to Google Drive
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f"File uploaded to Google Drive with ID: {file.get('id')}")