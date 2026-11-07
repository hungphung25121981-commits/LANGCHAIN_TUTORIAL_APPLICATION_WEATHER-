from dotenv import load_env

from langchain.chat_models import init_chat_model 
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
llm=ChatGoogleGenerativeAI(model='gemini-3.6-flash')
model=init_chat_model(llm)

message =
{
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image'},
         {'type': 'image', 'text': 'Describe the contents of this image'},
        ]

    
    
    
}