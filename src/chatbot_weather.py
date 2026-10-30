import requests, warnings
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import  ChatPromptTemplate, MessagesPlaceholder
from langchain.google_genai import ChatGoogleGenerativeAI

load_dotenv()


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
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant, who can always provide weather information for any location with humourous joke."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
        MessagesPlaceholder(variable_names="agent_scratchpad")
    ]
)

agent_create=create_agent(
    model=llm,
    tools=[get_weather],
    prompt=prompt)

agent_ex=AgentExecutor(agent=agent_create, tools=[get_weather], verbose=True)

print("Welcome to the Weather ChatBot! Type 'exit' to quit.")
while True:
    user_input=input("You: ")
    if user_input.lower()=='exit':
        print("Exiting the chat. Goodbye!")
        break
    else:
       
        response=agent_ex.invoke({
            'input': user_input,
            'chat_history': chat_history
        })
        if isinstance(response.content,list):
            bot_response=response.content[0]['text']
        else:
            bot_response=response.content
        print(f"Bot: {bot_response}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=bot_response))