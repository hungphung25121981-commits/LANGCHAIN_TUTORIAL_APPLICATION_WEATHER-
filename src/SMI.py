from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model='gemini-3.6-flash',
    model_provider='google_genai',
    temperature=1
)

response = model.invoke('Hello, how are you?')


if isinstance(response.content, list):
    print(response.content[0]['text'])
else:
    print(response.content)