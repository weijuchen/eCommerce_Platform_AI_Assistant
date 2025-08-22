from fastapi import FastAPI
from pydantic import BaseModel
from faq_rag import ask_question

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):
    answer = ask_question(query.question)
    return {"question": query.question, "answer": answer}
