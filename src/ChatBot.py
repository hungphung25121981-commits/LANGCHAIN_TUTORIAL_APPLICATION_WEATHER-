from multiprocessing import context
from urllib import response
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
import warnings
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*", category=UserWarning)
model=init_chat_model(
    model='gemini-3.6-flash',
    model_provider='google_genai',
    temperature=1
)

chat_history=[]

print("Welcome to the ChatBot dau tay! Type 'exit' to quit.")

while True:
    user_input=input("You: ")

    if user_input.lower()=='exit':
        print("Exiting the chat. Goodbye!")
        break
    chat_history.append(HumanMessage(content=user_input))
    context="You are a helpful assistant. "
    
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",context),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ])
    response=model.invoke(prompt.format_prompt(input=user_input,chat_history=chat_history).to_messages())
    if isinstance(response.content,list):
        bot_response=response.content[0]['text']
    else:
        bot_response=response.content
    print(f"Bot: {bot_response}")
    chat_history.append(AIMessage(content=bot_response))

