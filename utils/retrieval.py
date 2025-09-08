import faiss
import pickle

with open("qa_documents.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("qa_faiss_index.index")

def retrieve_documents(q_vec, top_k=3):
    distances, indices = index.search(q_vec, top_k)
    return [documents[i] for i in indices[0]]