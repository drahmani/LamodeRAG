import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline



# Load Q&A pkl and index from data

with open("qa_documents.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("qa_faiss_index.index")

#load embedding model 
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS(embedding_function=embed_model, index=index, docstore=None, index_to_docstore_id=None)


# now loading LLaMA-2 on GPU

model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit(smallest of all!) quantization to fit in VRAM
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
 # tokenisation 
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

#wrap in a  pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,# seems decent for now
    temperature=0.7,#standard
    do_sample=True,
    device=0  # for GPU
)

llm = HuggingFacePipeline(pipeline=pipe)


# build retrievalQA chain using Langchain

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)


# finally run a test 

query = "I am blonde with hazel eyes. I have a house warming party coming up, what should I wear?"
answer = qa.run(query)

print("Answer:", answer)
