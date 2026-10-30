
import requests, warnings
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import  ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")
llm=ChatGoogleGenerativeAI(model='gemini-3.6-flash')

def get_weather_func(location: str) -> str:
    try:
        response=requests.get(f'https://wttr.in/{location}?format=j1')
        return response.json()['current_condition'][0]['temp_C']+ "°C"
    except Exception:
        return "Chết òi, không tìm thấy thông tin thời tiết cho địa điểm này. Vui lòng thử lại lần sau "


get_weather=Tool(
    name='get_weather',
    description='Get the current weather for a given location including temperature, humidity, and wind speed.',
    func=get_weather_func,
    return_direct=False
    )

chat_history=[]

agent=create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant, who can always provide weather information for any location with humourous joke.")


print("Welcome to the Weather ChatBot! Type 'exit' to quit.")
while True:
    user_input=input("You: ")
    if user_input.lower()=='exit':
        print("Exiting the chat. Goodbye!")
        break
    else:
        chat_history.append(HumanMessage(content=user_input))
        response=agent.invoke({
            'messages': chat_history
            })
        final_message=response['messages'][-1]
        if isinstance(final_message.content, list):
            bot_response = final_message.content[0]['text']
        else:
            bot_response = final_message.content
        
        print(f"Bot: {bot_response}")
        
        chat_history.append(AIMessage(content=bot_response))