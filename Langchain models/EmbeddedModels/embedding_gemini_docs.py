from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',dimensions=32)

documents = [
    "Delhi is the capital of India",
    "I may get placed in GT",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))