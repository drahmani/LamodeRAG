from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_query(query):
    return embed_model.encode([query]).astype("float32")