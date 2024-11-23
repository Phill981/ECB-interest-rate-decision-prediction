import streamlit as st

def set_page_style():
    """
    Applies custom Streamlit page styles.
    """
    st.set_page_config(
        page_title="PDF Language Classifier",
        page_icon="📄",
        layout="centered"
    )
    st.markdown("""
        <style>
        body {
            background-color: #f9f9f9;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

def display_header():
    """
    Displays the app header with a title and description.
    """
    st.title("📄 PDF Language Classifier")
    st.markdown("""
        Analyze the language of PDF files and classify them based on similarity 
        with pre-labeled documents.
    """)
