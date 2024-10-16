import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from send_zoom_message import send_chat_to_zoom
from dotenv import load_dotenv

load_dotenv()

chat = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

conversation_history = {}

def call_groq_api(command_text, to_jid):
    try:
        new_user_prompt = f"Human: {command_text}\n\nAssistant:"

        prompt_template = PromptTemplate.from_template(
            """
            You are a helpful assistant. Respond to each user prompt as follows:
            ### HUMAN INPUT:
            {prompt}

            ### ASSISTANT:
            """
        )

        chain = prompt_template | chat
        res = chain.invoke(input={"prompt": new_user_prompt})

        completion = res.content.strip()

        conversation_history[to_jid] = new_user_prompt + completion

        return completion
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, something went wrong while processing your request."
