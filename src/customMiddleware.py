from dotenv import load_dotenv 
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest,ModelResponse, AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_cohere import ChatCohere
import time

load_dotenv()
# turtorial hướng dẫn

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time= 0.0 
    def before_agent(self,state: AgentState,runtime): 
        self.start_time= time()
        print("before_agent triggered")

        def before_model(self, state: AgentState, runtime): 
                print("before model triggered")
        def after_model(self, state: AgentState, runtime): 
            print("after model triggered")
        def after_agent(self, state: AgentState, runtime): 
            time_diff = time.time() - self.start_time 
            print(f"after agent: {time_diff}")


#code tay 

class SecurityandAuditmiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time=0
        self.call_count=0
    #chay khi user gui yeu cau
    def before_agent(self, state: AgentState,runtime):
        self.start_time=time.time()
        self.call_count=0

    # logic 1 block pword (hack and destroy)
        last_message=state["messages"][-1].content.lower()
        if "hack" in last_message or "destroy" in last_message:
            raise ValueError("Yeu cau bi tu choi do co tu khoa: ")
        print("[before agent] input hop le")

    # chay moi lan goi
    def before_model(self, state: AgentState, runtime):
        self.call_count+=1
        if self.call_count>5:
            raise RuntimeError("Goi 5 lan goi, nghi de!")
        print(f"[before model] chuan bi goi model lan {self.call_count}")

    # chay sau khi co response
    def after_model(self, state: AgentState,runtime):
        #logic  
        latest_response=state["messages"][-1]
        if hasattr(latest_response,"content") and isinstance(latest_response.content,str):
            if "nhạy cảm" in latest_response.content:
                latest_response.content= "Nội dung phản hồi đã bị ẩn vì vi phạm tiêu chuẩn cộng đồng "
            print(f"[after_model] kiem duyet phan hoi hoan tat ")
    
    def after_agent(self, state: AgentState,runtime):
        total_time=time.time()-self.start_time

        print(f"[after_agent] phien hoan tat! tong thoi gian {total_time:.2f}s")

llm=ChatCohere(model='command-r7b-12-2024')
agent=create_agent(
    model=llm,
    middleware=[SecurityandAuditmiddleware()],
    system_prompt="you are a helpful assistant")

user_input=input("Nhap pronpt: ")

response=agent.invoke(
    {
        'messages': [
                     HumanMessage(user_input)]
        })

print(response["messages"][-1].content)