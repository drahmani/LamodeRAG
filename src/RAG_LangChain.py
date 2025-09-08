import faiss
import pickle
import torch
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# -----------------------
# Load FAISS + Documents
# -----------------------
with open("qa_documents.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("qa_faiss_index.index")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS(embedding_function=embed_model, index=index, docstore=None, index_to_docstore_id=None)

# -----------------------
# Load LLaMA-2 on GPU
# -----------------------
model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Use 4-bit quantization to fit in 11GB VRAM
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Wrap in a HuggingFace pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    device=0  # ensure it runs on GPU
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------
# Build RetrievalQA chain
# -----------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# -----------------------
# Test a query
# -----------------------
query = "I am petite and brunette with blue eyes. I have a house warming party coming up, what should I wear?"
answer = qa.run(query)

print("Answer:", answer)
