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

def call_groq_api(command_text, to_jid):
    try:
        # Directly use the command_text as the prompt without history
        new_user_prompt = f"Human: {command_text}\n\nAssistant:"

        # Set up the prompt template for ChatGroq
        prompt_template = PromptTemplate.from_template(
            """
            You are a helpful assistant. Respond to each user prompt as follows:
            ### HUMAN INPUT:
            {prompt}

            ### ASSISTANT:
            """
        )

        # Send only the new prompt to ChatGroq
        chain = prompt_template | chat
        res = chain.invoke(input={"prompt": new_user_prompt})

        # Process the response
        completion = res.content.strip()

        # Optionally: update conversation history (if needed for future interactions)
        conversation_history[to_jid] = new_user_prompt + completion

        return completion
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, something went wrong while processing your request."
