from dataclasses import dataclass
from re import A
from dotenv import load_dotenv
import warnings 
from langchain_cohere import ChatCohere
from langchain.agents import create_agent 
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest,ModelResponse, wrap_model_call
from langchain_google_genai import ChatGoogleGenerativeAI as CG
from langchain_core.messages import HumanMessage, SystemMessage 


load_dotenv()
llm=CG(model='gemini-3.6-flash')
llm2=ChatCohere(model='command-r7b-12-2024')
basic_model=llm2 
advanced_model=llm 

# messages is list ( include: HumanMessage, AIMessage, ToolMessage, SystemMessage)
# trường dữ liệu state  có ( user_id, current_step, retry_count,summary)
'''
**request.state**

Chứa **dữ liệu động của phiên làm việc**, chủ yếu là danh sách tin nhắn (`state["messages"]`) và các biến state tích lũy qua các bước.

**request.model**

Đối tượng Chat Model (LLM) sẽ được gọi. Bạn có thể ghi đè biến này để đổi model.

**request.system\_prompt**

Câu lệnh hệ thống được gán riêng cho lượt gọi này (nếu có).

**request.tools**

Danh sách các công cụ (tools) đang được gắn vào mô hình trong lượt gọi.

**request.runtime**

Chứa ngữ cảnh runtime truyền từ ngoài vào (`request.runtime.context`), ví dụ như đối tượng `Context(user_role="child")`.
'''
@wrap_model_call
def dynamic_model_selection(request: ModelRequest,handler)->ModelResponse:
        message_count=len(request.state['messages'])
        
        if message_count > 3:
            print("Dang dung advanced_model\n")
            model=advanced_model 
        else:
            print("Dang dung basic model\n")
            model=basic_model

        request.model=model
        return handler(request)

agent=create_agent(model=basic_model, middleware=[dynamic_model_selection])

response=agent.invoke(
    {
        'messages': [ SystemMessage("You are a helpful assistant"),
                       HumanMessage("what is 1+1")]
        
        
        
        })


    

# Lấy danh sách tin nhắn an toàn (nếu không có key 'messages', mặc định trả về list rỗng [])
messages = response.get('messages', [])

# KIỂM TRA: Nếu có ít nhất 1 tin nhắn trong danh sách
if messages: 
    final_message = messages[-1]
    raw_content = final_message.content
    

    # Đặt mặc định answer là chuỗi gốc
    answer = str(raw_content) 

    if isinstance(raw_content, list):
        clean_text = [block.get('text', '') for block in raw_content if isinstance(block, dict)]
        if clean_text:
            answer = "\n".join(clean_text)
                
    elif isinstance(raw_content, str) and "'signature':" in raw_content:
        import ast
        try:
            parsed_dict = ast.literal_eval(raw_content)
            if isinstance(parsed_dict, dict) and 'text' in parsed_dict:
                answer = parsed_dict['text']
        except Exception:
            pass 

    print("\n🤖 AI Trả lời:")
    print(answer)
    
else:
    # Xử lý trường hợp bị rỗng để code không bị sập (crash)
    print("⚠️ Lỗi: Không nhận được tin nhắn phản hồi nào từ mô hình AI.")
    print("Dữ liệu thô trả về:", response)

