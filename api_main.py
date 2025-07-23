from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import logging
from embedding_generator import model, processor
from query_db import search_embeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
logger.info(f"Loading LLM model: {LLM_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, device_map="auto")
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
logger.info("LLM model loaded successfully.")

class QueryRequest(BaseModel):
    user_id: int
    query: str
    top_k: int = 2

@app.get("/")
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    logger.info(f"Received query for user_id {request.user_id}: {request.query}")
    try:
        batch_queries = processor.process_queries([request.query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
        query_embedding = query_embeddings.cpu().to(torch.float32).numpy()[0].tolist()
        results = search_embeddings(request.user_id, query_embedding, top_k=request.top_k)
        for r in results:
            if 'embedding' in r:
                r.pop('embedding')
        logger.info(f"Returning {len(results)} results for user_id {request.user_id}")
        top_context_1 = results[0]['content'] if len(results) > 0 else ""
        top_context_2 = results[1]['content'] if len(results) > 1 else ""
        prompt = f"""
You are a helpful assistant designed to answer questions for social media analysis.

Only use the provided context to answer the question.
If the answer is not present in the context, respond with:
"I’m sorry, I can’t answer that based on the current information."

User Query: {request.query}

Context:
1. {top_context_1}
2. {top_context_2}

Answer:
"""
        logger.info("Sending prompt to LLM.")
        llm_response = llm_pipe(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
        logger.info("LLM response generated.")
        return {"results": results, "llm_answer": llm_response}
    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")