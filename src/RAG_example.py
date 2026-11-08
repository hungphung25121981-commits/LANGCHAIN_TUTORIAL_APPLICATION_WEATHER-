from dotenv import load_dotenv
import os, warnings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from dataclasses import dataclass

load_dotenv()
warnings.filterwarnings("ignore",category=UserWarning,module="langchain_core")

embeddings= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
texts=[
    'Yêu một người không có ước mơ. Là một bài hài hay',
    'Apple là một công ty công nghệ nổi tiếng với các sản phẩm như iPhone, iPad và MacBook.',
    'Python và RUST là 2 ngôn ngữ lập trình phổ biến được sử dụng trong trí tuệ nhân tạo, 1 cái để nghiên cứu model, 1 cái để đẩy lên production',
    'Việt Nam là một quốc gia nằm ở Đông Nam Á, nổi tiếng với cảnh quan thiên nhiên tuyệt đẹp và văn hóa phong phú.',
    'Clair obscur expedition 33 là 1 tựa game hay về lối chơi turn-based, hướng tiếp cận mới cùng âm nhạc tuyệt vời ',
    'I love Apple. Là tôi yêu táo.'
]

vector_store=FAISS.from_texts(texts, embeddings)
retriever=vector_store.as_retriever(search_kwargs={'k':3})
retriever_tool=create_retriever_tool(retriever,name='kb_search',description='Search as the request with full consideration')



llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")


@dataclass
class User:
    userid: str


checkpoint=MemorySaver()
agent=create_react_agent(
    model=llm,
    tools=[retriever_tool],
    checkpointer=checkpoint
    )

def chat_AI(user:User, message: str)->str:
    config= {'configurable': {'thread_id': user.userid}}

    system=('You are a helpful assistant.For questions about things in the texts that are in vector_store,'
     'first call the kb_search tool to retrieve context, then answer succinctly.Rethink first by looking up at the context again before answering'
        )

    response=agent.invoke(
        {
            "messages":[
            SystemMessage(content=system),
            HumanMessage(content=message)
        ]},
        config=config
    )
    final_message=response['messages'][-1]
    raw_content=final_message.content

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



if __name__ == "__main__":
    print("         CHƯƠNG TRÌNH KHỞI ĐỘNG")
    
    # Đăng Nhập 
    u_id = input("Nhập user_id của bạn: ")
        

    current_user = User(userid=u_id)
    print(f"\n[Đã tạo phiên làm việc cho: {current_user.userid}]\n")
    
  
    while True:
        user_text = input("Bạn: ")
        
        if user_text.lower() == 'exit':
            print("Bot: Hẹn gặp lại bạn lần sau nha! Bái bai!")
            break
            
 
        bot_reply = chat_AI(user=current_user, message=user_text)
        
        print(f"Bot: {bot_reply}")
        print("-" * 30)