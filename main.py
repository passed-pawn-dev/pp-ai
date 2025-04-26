from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
import random
import os
from fastapi.middleware.cors import CORSMiddleware

if not os.environ.get("MISTRAL_API_KEY"):
  os.environ["MISTRAL_API_KEY"] = ""

# Sample documents
docs = [
    Document(page_content="User can register as a coach by pressing a green button on bottom left")
]

for i in range(20):
    nonsense = f"This is irrelevant content #{i}: {random.choice(['Cats dance on Mars.', 'Bananas talk philosophy.', 'Llamas run programming bootcamps.', 'Umbrellas are political.', 'Blue cheese unlocks portals.'])}"
    docs.append(Document(page_content=nonsense))

# Embeddings and retriever
embeddings = MistralAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt(inputs):
    return f"""You are a helpful assistant. Answer in a very friendly way. You are a mascot elephant.
If it is not given in the context, say banana.
Answer the question based only on the context below:

Context:
{inputs['context']}

Question: {inputs['question']}
"""

# LLM and chain
llm = ChatMistralAI(model="mistral-large-latest")
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | RunnableLambda(build_prompt)
    | llm
    | StrOutputParser()
)

# FastAPI app
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}
