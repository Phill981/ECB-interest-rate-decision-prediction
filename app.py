import os
import streamlit as st
from models.pdf_processor import PDFProcessor
from models.classifier import PDFClassifier
from styles import set_page_style, display_header

# Initialize components
pdf_processor = PDFProcessor()
classifier = PDFClassifier()
labeled_pdf_dir = "labeled_pdfs"

# Load pre-labeled PDFs
if os.path.exists(labeled_pdf_dir):
    classifier.load_labeled_pdfs(labeled_pdf_dir, pdf_processor)
else:
    raise FileNotFoundError("Labeled PDFs directory not found!")

# Set page styles
set_page_style()
display_header()

# Upload PDF section
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = "temp_uploaded_file.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Perform classification
    classification_result = classifier.classify_pdf(temp_file_path, pdf_processor)

    # Display results
    st.subheader("Classification Result")
    st.write(f"The uploaded PDF is classified as: **{classification_result[1]}**")
    st.write(f"Similarity Score: **{classification_result[0]:.4f}**")
    
    # Clean up temporary file
    os.remove(temp_file_path)
else:
    st.info("Please upload a PDF file for analysis.")
