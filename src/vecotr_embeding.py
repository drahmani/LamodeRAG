import json
from sentence_transformers import SentenceTransformer
import faiss #FB AI similarity Search
import numpy as np
import pickle
# Load Q&A JSON
with open("fashion_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Combine Q&A into a single "document" per entry
documents = [f"Q: {item['Question']} A: {item['Answer']}" for item in qa_data]

#load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#Generate embedding
embeddings = model.encode(documents, show_progress_bar = True)#list of vectors size 384

#store them in faiss vector store

embeddings = np.array(embeddings).astype("float32")
#create faiss index

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) #L2 distance save all the vectors by flattening them
index.add(embeddings)
print(f"Number of vectors in FAISS: {index.ntotal}")

faiss.write_index(index, "qa_faiss_index.index")

# Optionally save the raw documents for retrieval

with open("qa_documents.pkl", "wb") as f:
    pickle.dump(documents, f)
