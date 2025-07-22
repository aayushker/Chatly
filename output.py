from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = f"""
You are a helpful assistant designed to answer questions for social media analysis.

Only use the provided context to answer the question.
If the answer is not present in the context, respond with:
"I’m sorry, I can’t answer that based on the current information."

User Query: {user_query}

Context:
1. {top_context_1}
2. {top_context_2}

Answer:
"""

response = pipe(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
print(response)
