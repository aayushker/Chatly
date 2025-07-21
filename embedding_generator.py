import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

model_name = "nomic-ai/nomic-embed-multimodal-3b"
model = BiQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = BiQwen2_5_Processor.from_pretrained(model_name)

queries = [
    "Brand X launched a new sustainable campaign which received positive engagement across platforms.",
    "Users criticized Brand Y for lack of transparency in recent posts.",
]

batch_queries = processor.process_queries(queries).to(model.device)

with torch.no_grad():
    query_embeddings = model(**batch_queries) 

print(query_embeddings.shape)  

embeddings_np = query_embeddings.cpu().to(torch.float32).numpy()