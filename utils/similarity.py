from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Calculate Cosine Similarity
def calculate_cosine_similarity(text, labeled_documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embeddings for the input text and labeled documents
    text_embedding = model.encode([text])[0]
    labeled_embeddings = model.encode(labeled_documents)

    # Calculate similarities
    similarities = cosine_similarity([text_embedding], labeled_embeddings).flatten()

    # Return similarities and their corresponding labels
    return [{"label": labeled_documents[i], "similarity": similarities[i]} for i in range(len(labeled_documents))]


