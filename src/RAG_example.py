from dotenv import load_dotenv
import os, warnings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

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
results = vector_store.similarity_search("Hoa Quả nào mà người đó thích nhất", k=3)
answers = [doc.page_content for doc in results]

print(answers)