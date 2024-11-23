import pdfplumber

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extracts text from a given PDF file.
        """
        with pdfplumber.open(pdf_path) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
