import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

load_dotenv()

# Define Context structure
@dataclass
class Context:
    user_id: str 

# Define structured output format (TypedDict is often easier for tool calling/structured output)
class ResponseFormat(TypedDict):
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

# Initialize the LLM (using a current valid Gemini model)
llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

# 1. Weather Tool
@tool
def get_weather(location: str) -> str :
    """Get the current weather for a given location."""
    try:
        response = requests.get(f'https://wttr.in/{location}?format=j1')
        return response.json()['current_condition'][0]['temp_C'] + "°C"
    except Exception:
        return "Chết òi, không tìm thấy thông tin thời tiết cho địa điểm này. Vui lòng thử lại lần sau."

# 2. Location Tool (Simulating context lookup) 
@tool
def get_location(user_id: str) -> str:
    """Get the current location of the user based on their user_id."""
    match user_id:
        case 'ABC123':
            return "Vienna"
        case 'XYZ789':
            return "New York"
        case 'LMN456':
            return "Tokyo"
        case _: 
            return "Unknown location"

# Initialize memory for the agent
checkpointer = MemorySaver()

# System prompt
a = "You are a helpful assistant who provides weather information for a user's location with a humorous joke. First, find their location using their user_id, then get the weather."

# Create the agent using LangGraph
agent = create_agent(
    model=llm,
    tools=[get_location, get_weather],
    system_prompt=a,
    checkpointer=checkpointer
)

# Configuration and input
config = {'configurable': {'thread_id': '1'}}
content_user = "What is the weather like where I am?" # Added the missing input variable
current_user_id = 'ABC123'

# Invoke the agent
response = agent.invoke(
    {
        "messages": [
            HumanMessage(content=f"My user_id is {current_user_id}. {content_user}")
        ]
    },
    config=config
)

# Extract and print the final response
final_message = response['messages'][-1]

print(final_message.content)
       


