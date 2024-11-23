import streamlit as st
import os
from utils.similarity import calculate_cosine_similarity
import matplotlib.pyplot as plt

# Your other Streamlit app code
def main():
    st.title("ECB interest rate decision prediction")

    # File upload and processing
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)  # You should define this function to extract text from the PDF

        # Labeled documents for comparison (you can replace this with your actual labeled documents)
        labeled_documents = [
            "Monetary policy decisions increase",
            "Monetary policy decisions decrease",
            "Economic contraction policy increase",
            "Economic expansion policy decrease"
        ]
        
        # Calculate cosine similarities
        similarity_results = calculate_cosine_similarity(text, labeled_documents)

        # Display similarity results
        st.write("Cosine Similarity Scores:")
        for result in similarity_results:
            st.write(f"{result['label']}, Similarity: {result['similarity']:.4f}")

        # Plot the similarity chart
        plot_similarity_chart(similarity_results)


# Plot Cosine Similarity Scores
import plotly.graph_objects as go
import plotly.express as px

# Plot Cosine Similarity Scores with Plotly (Interactive & Colorful)
def plot_similarity_chart(similarity_results):
    labels = [result["label"] for result in similarity_results]
    scores = [result["similarity"] for result in similarity_results]

    # Create a Plotly bar chart
    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',  # Horizontal bars
        marker=dict(
            color=scores,  # Color by score
            colorscale='Viridis',  # A colorful scale
            colorbar=dict(title='Cosine Similarity')
        ),
        text=[f"{score:.4f}" for score in scores],  # Show score as text on hover
        hoverinfo='text',  # Display score on hover
    ))

    # Update the layout to improve aesthetics and interactivity
    fig.update_layout(
        title='Similarity Comparison',
        xaxis_title='Cosine Similarity Score',
        yaxis_title='Document Labels',
        title_font=dict(size=20, family='Arial', color='#333333'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)'),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',  # Background color of the plot
        template='plotly_white',  # Use Plotly's clean white template
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)



# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    # You can use PyPDF2 or pdfplumber to extract text from the PDF
    import pdfplumber
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


if __name__ == "__main__":
    main()
