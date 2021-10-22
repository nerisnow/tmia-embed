
import logging

import pandas as pd

from fastapi import FastAPI
from fastapi.logger import logger
from mangum import Mangum
from numpy import load
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


gunicorn_logger = logging.getLogger("gunicorn.error")
logger.handlers = gunicorn_logger.handlers
logger.setLevel(logging.DEBUG)


class Query(BaseModel):
    query: str
    # intent: Optional[str]


app = FastAPI()

model = SentenceTransformer(model_name_or_path="../models")

@app.get("/")
async def home():
    return {"app": "Suggestive Questions"}


@app.post("/embed")
async def query(q: Query):
   corpus_embedding = model.encode(q.query)
   res = {
       "question_vector": corpus_embedding.tolist()
   }
   return res


handler = Mangum(app)