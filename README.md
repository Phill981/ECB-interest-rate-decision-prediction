# ECB interest rate decision prediction

This program analyses ECB statements that are labled depending if somewhen afterwards interest rates increased or decreased. Based on language similarities, the model will then predict the next ECB decision regarding interest rates.

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
