# Fashion RAG

A domain-specific Retrieval-Augmented Generation (RAG) pipeline for fashion Q&A.  
Uses pre-trained LLMs (LLaMA / Mistral) and FAISS for document retrieval.

Project Structure:
- src/        : Python scripts (RAG pipeline, LangChain, scraping)
- data/       : Processed data, FAISS index, PKL files
- configs/    : YAML configuration files
- notebooks/  : Optional Jupyter notebooks for exploration

Features:
- Retrieve relevant fashion Q&A examples from my dataset
- Generate context-aware answers using llama and Mistral instruct models on GPU 
- Also using LangChain integration for comparison the RAG workflows
- will be trying quantization  for large models

Installation:
1. Clone the repository: https://github.com/drahmani/LamodeRAG

git clone 
cd FashionRAG

2. Install dependencies:

pip install -r requirements.txt


Usage:

python src/RAG_gpu.py


Modify files in `configs/` or `data/` to customize prompts or datasets.

Contributing:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

