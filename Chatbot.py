from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import faiss
import numpy as np
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# Load FAISS index and chunks
index = faiss.read_index("faiss_index.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# OpenAI client
client = OpenAI(api_key="Add your api key............")

# Helper functions
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def retrieve_chunks(query, k=3):
    query_vec = embed_text(query).reshape(1, -1)
    scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def chatbot(query):
    retrieved = retrieve_chunks(query, k=2)
    context = "\n".join(retrieved)
    prompt = f"""
You are an expert insurance assistant bot.
Use the following knowledge base to answer the user's question:

Knowledge base:
{context}

User question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# FastAPI app
app = FastAPI()

# Allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your website domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    answer = chatbot(query.question)
    return {"answer": answer}
