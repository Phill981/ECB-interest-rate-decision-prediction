from sentence_transformers import SentenceTransformer, util
import os
import glob

class PDFClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the Hugging Face model for embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.labeled_pdfs = []

    def load_labeled_pdfs(self, pdf_directory, pdf_processor):
        """
        Loads labeled PDFs from the specified directory and computes embeddings.
        """
        for pdf_file in glob.glob(f"{pdf_directory}/*.pdf"):
            label = "increase" if "increase" in pdf_file.lower() else "decrease"
            text = pdf_processor.extract_text_from_pdf(pdf_file)
            embedding = self.model.encode(text, convert_to_tensor=True)
            self.labeled_pdfs.append({"file": pdf_file, "label": label, "embedding": embedding})

    def classify_pdf(self, uploaded_pdf_path, pdf_processor):
        """
        Classifies the uploaded PDF based on cosine similarity with labeled PDFs.
        """
        uploaded_text = pdf_processor.extract_text_from_pdf(uploaded_pdf_path)
        uploaded_embedding = self.model.encode(uploaded_text, convert_to_tensor=True)
        
        similarities = [
            (util.pytorch_cos_sim(uploaded_embedding, labeled_pdf["embedding"]).item(), labeled_pdf["label"])
            for labeled_pdf in self.labeled_pdfs
        ]
        
        best_match = max(similarities, key=lambda x: x[0])
        return best_match
