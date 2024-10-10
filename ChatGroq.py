import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from send_zoom_message import send_chat_to_zoom
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize ChatGroq with your API key and model
chat = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

# Keep conversation history to maintain context with the user
conversation_history = {}

def call_groq_api(payload):
    try:
        user_jid = payload['toJid']
        command_text = payload.get('cmd', '')

        # Retrieve the conversation history for the user
        history = conversation_history.get(user_jid, "")
        new_user_prompt = f"\n\nHuman: {command_text}\n\nAssistant:"
        prompt = history + new_user_prompt

        # Create a prompt template for a more interactive response
        prompt_template = PromptTemplate.from_template(
            """
            You are a helpful assistant. Respond to each user prompt based on the conversation history.
            ### HUMAN INPUT:
            {prompt}
            
            ### ASSISTANT:
            """
        )

        # Send the user prompt to ChatGroq and get a response
        chain = prompt_template | chat
        res = chain.invoke(input={"prompt": prompt})

        # Extract the generated response from the model
        completion = res.content.strip()

        # Update conversation history for context in future interactions
        conversation_history[user_jid] = prompt + completion

        # Send the response back to Zoom chat
        send_chat_to_zoom(completion, payload)

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Optional: Send an error message back to Zoom
        error_message = "Sorry, something went wrong while processing your request."
        send_chat_to_zoom(error_message, payload)

