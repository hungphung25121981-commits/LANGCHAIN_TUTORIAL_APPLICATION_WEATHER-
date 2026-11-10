
from dotenv import load_dotenv
import warnings

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
#ModelResponse-ModelRequest； Request la đối tượng đóng gói toàn bộ thông tin; Chứa thông tin chuẩn bị dc gửi đến
#Response: đóng gói toàn bộ kết quả AI trả về 
# hai thằng đều giúp chặn thông tin để kiểm duyệt (Request là từ user, response là từ AI )

from dataclasses import dataclass
load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')


@dataclass
class Context:
    user_role: str


''' 
@dynamic_prompt tao 1 lớp prompt động 
có nghĩa là trước thì system_prompt sẽ khóa cứng 1 prompt còn thêm dymastic có thể thay đổi câu lệnh theo runtime 
khi có 1 yêu cầu mới đến dynamic sẽ xử lý ngay nút này sau đó tạo ra dựa trên các logics mà ng tạo đưa ra ( như match case)
'''
@dynamic_prompt
def user_role_prompt(request: ModelRequest)->str:
    user_role= request.runtime.context.user_role 

    base_prompt= 'You are a helpful assistant.'

    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detailed technical responses'
        case 'beginner':
            return f'{base_prompt} Keep your explanations simple and basic'
        case 'child':
            return f'{base_prompt} Explain everything as if you were talking to a five-year child'
        case _:
            return base_prompt 
agent= create_agent(
    model=llm,
    middleware=[user_role_prompt],
    context_schema=Context
)


for user in ['beginner','child','expert']:
 response=agent.invoke(
    {
        'messages': [{'role': 'user',
                      'content': 'Explain PCA'
                      }]
     },
     context=Context(user_role=user)
 )
 if isinstance(response, dict) and 'output' in response:
        print(response['output'])
        
    # Hoặc lấy nội dung nếu nó là một đối tượng AIMessage
 elif hasattr(response, 'content'):
        print(response.content)
        
    # Phòng hờ trường hợp khác
 else:
        print(response)
        
 print("===================================================\n")
