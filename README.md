# Fashion RAG

This repo contains code for Fashion RAG a development github to give users advice about clothing given the individuals preferences, time of the year, weather and possibly user input event information. This github currently contains sample question and answer exemplars for augmented queries. (Image analysis using Yolo to extract features has also been researched but is not included here as performance is not satisfactory yet). 

RAG pipeline for fashion color analysis Q&A using pre-trained LLMs (LLaMA/Mistral and LangChain) and FAISS for document retrieval.

Project Structure:
- src/        : Python scripts (RAG pipeline, LangChain, scraping)
- data/       : Processed data, FAISS index, PKL files
- configs/    : YAML configuration files
- notebooks/  : empty
- utils/ (not used currently)

Features:
- Retrieve relevant fashion Q&A examples from my dataset
- Generate context-aware answers using llama and Mistral instruct models on GPU 
- Also using LangChain integration for comparison the RAG workflows
- will be trying quantization for fine tuning models later 

Installation:
1. Clone the repository: 

git clone https://github.com/drahmani/LamodeRAG
cd FashionRAG

2. Create a new Conda environment with Python 3.10:

    conda create --name RAG_gpu python=3.10
    conda activate RAG_gpu

3. Install dependencies:

pip install -r requirements.txt

4. **Evaluation Example:**
   

   python src/RAG_gpu.py

Modify files in `configs/` or `data/` to customize prompts or datasets.

Contributing:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

