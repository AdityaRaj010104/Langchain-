from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model ='models/gemini-embedding-001',dimensions = 300 )

document = [
    "Virat Kohli is known for his aggressive batting and unmatched consistency.",
    "Rohit Sharma is famous for his elegant stroke play and multiple double centuries in ODIs.",
    "MS Dhoni is celebrated for his calm leadership and finishing skills.",
    "Jasprit Bumrah is renowned for his lethal yorkers and unorthodox bowling action.",
    "Ravindra Jadeja is valued for his all-round abilities in batting, bowling, and fielding."
]

query = 'tell me about dhoni'

doc_embeddings = embedding.embed_documents(document)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(document[index])
print("Smilarity cosine is",score)