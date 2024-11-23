import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

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


