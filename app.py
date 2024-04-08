import streamlit as st
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

# Function to generate word embeddings and visualize
def generate_embeddings(text):
    # Remove stopwords
    text = remove_stopwords(text)
    # Tokenize the text
    tokenized_text = [sentence.split() for sentence in text.split('.')]
    # Train Word2Vec model
    model = Word2Vec(tokenized_text, min_count=1)

    # Extracting words from the text
    words = [word for word in text.split() if len(word) > 1 and word in model.wv]

    # Get word vectors for the extracted words
    word_vectors = model.wv[words]

    # Create DataFrame for plotting
    embeddings_df = pd.DataFrame(word_vectors, columns=[f'x{i}' for i in range(word_vectors.shape[1])])
    embeddings_df['word'] = words

    # Plot the embeddings using clusters of similar words
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot clusters of similar words
    for word in words:
        try:
            similar_words = [word] + [similar_word for similar_word, _ in model.wv.most_similar(word, topn=4) if similar_word in words]
            embeddings = model.wv[similar_words]
            ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
            for sim_word, embd in zip(similar_words, embeddings):
                ax.text(embd[0], embd[1], sim_word, fontsize=8)
        except KeyError:
            st.warning(f"No similar words found for '{word}'.")
        ax.set_title('Cluster words')

    st.pyplot(fig)

    return model

# Streamlit UI
st.title('Word Embeddings Visualization')

text = st.text_area('Enter text:')
if st.button('Generate Embeddings'):
    model = generate_embeddings(text)
