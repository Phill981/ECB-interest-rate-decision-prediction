from sentence_transformers import SentenceTransformer #type: ignore
from sklearn.metrics.pairwise import cosine_similarity #type: ignore

def calculate_cosine_similarity(text, labeled_documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text_embedding = model.encode([text])[0]
    labeled_embeddings = model.encode(labeled_documents)

    similarities = cosine_similarity([text_embedding], labeled_embeddings).flatten()

    return [{"label": labeled_documents[i], "similarity": similarities[i]} for i in range(len(labeled_documents))]
