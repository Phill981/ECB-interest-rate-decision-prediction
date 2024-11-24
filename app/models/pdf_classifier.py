from sentence_transformers import SentenceTransformer #type: ignore
from utils.similarity import cosine_similarity #type: ignore
from settings import Settings #type: ignore

class PDFClassifier:
    def __init__(self, model_name=Settings.model):
        self.model = SentenceTransformer(model_name)
        self.labeled_documents = self.load_labeled_documents()
        self.labeled_embeddings = self.compute_labeled_embeddings()

    def load_labeled_documents(self):
        labeled_documents = [
            {"text": "Monetary policy decisions", "label": "increase"},
            {"text": "Economic contraction policy", "label": "decrease"}
        ]
        return labeled_documents

    def compute_labeled_embeddings(self):
        labeled_texts = [doc["text"] for doc in self.labeled_documents]
        return self.model.encode(labeled_texts)

    def classify_pdf(self, file_path, pdf_processor):
        text = pdf_processor.extract_text(file_path)

        text_embedding = self.model.encode([text])[0]

        best_match = self.get_best_match(text_embedding)
        return best_match

    def get_best_match(self, text_embedding):
        best_similarity = 0
        best_label = ""
        for i, doc in enumerate(self.labeled_documents):
            labeled_embedding = self.labeled_embeddings[i]
            similarity = cosine_similarity(text_embedding, labeled_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_label = doc["label"]

        return best_similarity, best_label

    def get_labeled_documents(self):
        return [doc["text"] for doc in self.labeled_documents]
