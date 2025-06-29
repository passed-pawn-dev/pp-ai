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
# docs = [
#     Document(page_content="User can register as a coach by pressing a green button on bottom left")
# ]

# for i in range(20):
#     nonsense = f"This is irrelevant content #{i}: {random.choice(['Cats dance on Mars.', 'Bananas talk philosophy.', 'Llamas run programming bootcamps.', 'Umbrellas are political.', 'Blue cheese unlocks portals.'])}"
#     docs.append(Document(page_content=nonsense))

#Load documents
def load_documents_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [Document(page_content=line.strip()) for line in lines if line.strip()]

student_docs = load_documents_from_txt("documents/student.txt")
coach_docs = load_documents_from_txt("documents/coach.txt")
unregistered_docs = load_documents_from_txt("documents/unregistered.txt")

# Embeddings and retriever
embeddings = MistralAIEmbeddings()

student_retriever = Chroma.from_documents(documents=student_docs, embedding=embeddings, collection_name="student").as_retriever()
coach_retriever = Chroma.from_documents(documents=coach_docs, embedding=embeddings, collection_name="coach").as_retriever()
unregistered_retriever = Chroma.from_documents(documents=unregistered_docs, embedding=embeddings, collection_name="unregistered").as_retriever()

# Helper functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt(inputs):
    return f"""You are a helpful assistant. Answer in a very friendly way. You are a mascot elephant.
Answer the question based only on the context below.
If it is not given in the context, refer to the faq page.

Context:
{inputs['context']}

Question: {inputs['question']}
"""

# LLM and chain
llm = ChatMistralAI(model="mistral-large-latest")

rag_chains = {
    "student": ({
        "context": student_retriever | format_docs,
        "question": RunnablePassthrough()
    } | RunnableLambda(build_prompt) | llm | StrOutputParser()),

    "coach": ({
        "context": coach_retriever | format_docs,
        "question": RunnablePassthrough()
    } | RunnableLambda(build_prompt) | llm | StrOutputParser()),

    "unregistered": ({
        "context": unregistered_retriever | format_docs,
        "question": RunnablePassthrough()
    } | RunnableLambda(build_prompt) | llm | StrOutputParser())
    }

# FastAPI app
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest, role: str):
    if role not in rag_chains:
        return {"error": "Invalid role. Use one of: student, coach, unregistered"}

    answer = rag_chains[role].invoke(request.question)
    return {"answer": answer}
