import requests
from dotenv import load_dotenv
from langchain.agents import create_agent 
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

# 1. Khai báo hàm bình thường (KHÔNG CÓ @ ở trên)
def get_weather_func(location: str) -> str:
    response = requests.get(f'https://wttr.in/{location}?format=j1')
    return response.json()['current_condition'][0]['temp_C']

# 2. Tạo đối tượng Tool và truyền hàm vào func
get_weather = Tool(
    name='get_weather',
    description='Get the current weather for a given location',
    func=get_weather_func, # Truyền hàm vào đây!
    return_direct=False
)







agent=create_agent(
    model=llm,
    tools= [get_weather],
    system_prompt='You are a helpful assistant, who can always provide weather information for any location with humourous joke.'
    )
location=input("Enter a location to get the current weather: ")
content_user="What iss the current weather in " + location + "?"
response= agent.invoke({
    'messages': [
        {
            'role': 'user',
            'content': content_user
            }]
    })

final_message = response['messages'][-1]
final_content = final_message.content

# Nếu Gemini trả về cấu trúc danh sách khối (List of Blocks)
if isinstance(final_content, list):
    # Trích xuất giá trị 'text' từ khối đầu tiên
    print(final_content[0]['text'])
# Nếu Gemini trả về chuỗi văn bản bình thường (String)
else:
    print(final_content)
