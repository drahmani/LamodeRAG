# ===== RAG with local LLaMA-2-7B + ccompute timing in GPU + #token  

import os
import time
import pickle
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

#silence tokenizer  warnings so many of them!
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#give paths of dl models llama and or mistrel 
#model_path = os.path.expanduser("~/LLM/llama/hf-llama-2-7b")  # local LLaMA-2-7B
model_path = os.path.expanduser("~/LLM/mistral/mistral-7b-instruct")  # local Mistral

#ggive the path for generated Q&A samples as context for model to use for retrival
index_path = "qa_faiss_index.index"
docs_path = "qa_documents.pkl"

#embedding model 
print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# load Q&A index and  docs 
print("Loading FAISS index and documents...")
index = faiss.read_index(index_path)
with open(docs_path, "rb") as f:
    documents = pickle.load(f)

# for now I want to load tokenizer and  model
print("Loading Mistral model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    low_cpu_mem_usage=True
)
# setting for GPU memory check after loading ---crashed few times
if torch.cuda.is_available():
    print(f"\n GPU Memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated / {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

#RAG
# 1- retrieval step--------
def retrieve_documents(query, top_k=3):
    q_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    return [documents[i] for i in indices[0]]

# 2- Generation (RAG) --------
def generate_answer(query, max_new_tokens=200):
    # now start timing
    start_time = time.time()

    # and retrieve docs
    retrieved = retrieve_documents(query)
    print("\n Retrieved context:")
    for i, doc in enumerate(retrieved, 1):
          snippet = doc[:200] if isinstance(doc, str) else str(doc)
          print(f"  {i}. {snippet}...")


    context = "\n".join([f"Example {i+1}: {doc}" for i, doc in enumerate(retrieved)])

    # build specific prompt
    prompt = f"""
You are a professional fashion stylist. 
Use the context if it is useful. If the context is missing info, rely on your own knowledge of fashion.
If the answer truly cannot be given, reply: "I don’t know."

Context (fashion Q&A examples):
{context}

User Question:
{query}

Answer (as a fashion stylist):
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    #count number of tokens used
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"\n Prompt token count: {prompt_tokens}")

    # pshing inputs to GPU (I have it!!)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # generate  answers based on the retreived answers!!
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,   # deterministic, no randomness
            do_sample=False
        )

    total_tokens = output.shape[1]
    generated_tokens = total_tokens - prompt_tokens
    print(f" Generated token count: {generated_tokens}")
    print(f" Total tokens (prompt + output): {total_tokens}")

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split("Answer (as a fashion stylist):")[-1].strip()

    # end of calculating the timing
    elapsed = time.time() - start_time
    print(f"\n Answer generated in {elapsed:.2f} seconds")

      # --- GPU memory check after generation ---
    if torch.cuda.is_available():
        print(f" GPU Memory after generation: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated / {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")


    return answer

# run a test  --------
if __name__ == "__main__":
    query = "what top and jeans combination fits my petite figure?" 
    answer = generate_answer(query, max_new_tokens=120)
    print("\nFinal Answer:\n", answer)

    # cleanup to empty cach for gpu
    del model
    del tokenizer
    torch.cuda.empty_cache()
