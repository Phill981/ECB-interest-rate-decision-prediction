# PDF Classification and Similarity Comparison App

This Streamlit app classifies PDFs based on semantic similarity using pre-trained transformer models. It extracts text from PDFs and compares them to labeled datasets to classify and evaluate their similarity.

### Features:
- Upload multiple PDFs and compare their content with labeled PDFs
- Displays a summary of each uploaded document
- Visualizes cosine similarity scores in a bar chart
- Option to download the results as a CSV file

### Requirements:
- Python 3.7+
- Streamlit
- Sentence-Transformers
- PDFPlumber
- Torch

To run the app, simply install the requirements and start the Streamlit server:

```bash
pip install -r requirements.txt
streamlit run app.py


### How to Run the App:
1. Clone the repository or create the files in your project.
2. Install the dependencies by running:
   ```bash
   pip install -r requirements.txt
