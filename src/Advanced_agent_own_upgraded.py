import requests, warnings 
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
# 1. Định nghĩa Class gom toàn bộ thông tin đăng nhập
@dataclass
class Context:
    user_id: str 
    user_locate: str = None 

# 2. Định nghĩa cấu trúc trả về cho Tool
class ResponseFormat(TypedDict):
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

# 4. Weather Tool
@tool
def get_weather(location: str) -> ResponseFormat:
    """Get the current weather for a given location."""
    try:
       
        response = requests.get(f'https://wttr.in/{location}?format=j1').json()
        current_cond = response['current_condition'][0]
        
        return {
            "summary": current_cond['weatherDesc'][0]['value'],
            "temperature_celsius": float(current_cond['temp_C']),
            "temperature_fahrenheit": float(current_cond['temp_F']),
            "humidity": float(current_cond['humidity'])
        }
    except Exception:
        return {
            "summary": "Chết òi, không tìm thấy thông tin thời tiết cho địa điểm này. Vui lòng thử lại lần sau.",
            "temperature_celsius": 0.0,
            "temperature_fahrenheit": 0.0,
            "humidity": 0.0
        }

# 5. Khởi tạo Agent + mem 
checkpoint = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpoint
)

# 6. Hàm xử lý Chat 
def chat_AI(user_context: Context, message: str) -> str:
    # Lấy thread_id từ object
    config = {'configurable': {'thread_id': user_context.user_id}}
    
    # Bơm thông tin từ object vào Prompt
    if user_context.user_locate: 
        context_info = f"Người dùng có user_id {user_context.user_id} đang ở {user_context.user_locate}. Hãy cung cấp thông tin thời tiết cho họ."
    else:
        context_info = f"Người dùng có user_id {user_context.user_id} không cung cấp vị trí. Hãy hỏi họ vị trí trước khi cung cấp thông tin thời tiết."
        
    system = f"Thông Tin Ngữ Cảnh: {context_info}. Bạn là 1 trợ lý có ích, có khả năng cung cấp các thông tin chính xác, kèm khả năng chơi chữ thượng thừa. LƯU Ý: HỎI HỌ MUỐN LẤY THÔNG TIN Ở ĐÂU (MẶC ĐỊNH LÀ Ở LOCATION CỦA HỌ), KHI HỌ NÓI LẤY CHỖ KHÁC THÌ KẾT NỐI TỚI ĐÓ."
    
    # Gửi qua LangGraph
    response = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system),
                HumanMessage(content=message)
            ]
        },
        config=config
    )

    final_message = response['messages'][-1]
    raw_content = final_message.content
    

    if isinstance(raw_content, list):
        clean_text = [block['text'] for block in raw_content if isinstance(block, dict) and 'text' in block]
        return "\n".join(clean_text) if clean_text else str(raw_content)
 
    elif isinstance(raw_content, str) and "'signature':" in raw_content:
        import ast
        try:
            
            parsed_dict = ast.literal_eval(raw_content)
            if isinstance(parsed_dict, dict) and 'text' in parsed_dict:
                return parsed_dict['text']
        except Exception:
            pass 

    return str(raw_content)

# demo 
if __name__ == "__main__":
    print("         CHƯƠNG TRÌNH KHỞI ĐỘNG")
    
    # Đăng Nhập 
    u_id = input("Nhập user_id của bạn: ")
    u_loc = input("Nhập địa điểm hiện tại của bạn (Nhấn Enter để bỏ qua): ")
    if not u_loc.strip():
        u_loc = None 
        

    current_user = Context(user_id=u_id, user_locate=u_loc)
    print(f"\n[Đã tạo phiên làm việc cho: {current_user.user_id}]\n")
    
  
    while True:
        user_text = input("Bạn: ")
        
        if user_text.lower() == 'exit':
            print("Bot: Hẹn gặp lại bạn lần sau nha! Bái bai!")
            break
            
 
        bot_reply = chat_AI(user_context=current_user, message=user_text)
        
        print(f"Bot: {bot_reply}")
        print("-" * 30)