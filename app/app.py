import streamlit as st#type: ignore
from utils.similarity import calculate_cosine_similarity#type: ignore
import plotly.graph_objects as go #type: ignore

def main():
    st.title("ECB interest rate decision prediction")

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)

        labeled_documents = [
            "Monetary policy decisions increase",
            "Monetary policy decisions decrease",
            "Economic contraction policy increase",
            "Economic expansion policy decrease"
        ]
        
        similarity_results = calculate_cosine_similarity(text, labeled_documents)

        st.write("Cosine Similarity Scores:")
        for result in similarity_results:
            st.write(f"{result['label']}, Similarity: {result['similarity']:.4f}")

        plot_similarity_chart(similarity_results)

def plot_similarity_chart(similarity_results):
    labels = [result["label"] for result in similarity_results]
    scores = [result["similarity"] for result in similarity_results]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h', 
        marker=dict(
            color=scores,
            colorscale='Viridis',
            colorbar=dict(title='Cosine Similarity')
        ),
        text=[f"{score:.4f}" for score in scores],
        hoverinfo='text', 
    ))

    fig.update_layout(
        title='Similarity Comparison',
        xaxis_title='Cosine Similarity Score',
        yaxis_title='Document Labels',
        title_font=dict(size=20, family='Arial', color='#333333'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',  
        template='plotly_white',  
        margin=dict(l=50, r=50, t=50, b=50),  
    )
    st.plotly_chart(fig)

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    import pdfplumber #type: ignore
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    main()
