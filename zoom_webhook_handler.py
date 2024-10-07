from ChatGroq import call_groq_api
from flask import jsonify# Updated to use ChatGroq

def handle_zoom_webhook(request):
    try:
        event = request.json.get('event')
        
        if event == 'bot_notification':
            print('Zoom Team Chat App message received.')
            payload = request.json.get('payload')
            call_groq_api(payload)  # Pass the message to ChatGroq for processing
        elif event == 'bot_installed':
            print('Zoom for Team Chat installed.')
        elif event == 'app_deauthorized':
            print('Zoom for Team Chat uninstalled.')
        else:
            print(f'Unsupported event type: {event}')

        return jsonify({"message": "Event processed."}), 200
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500
