import streamlit as st
import numpy as np
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import time

# Download NLTK resources only if not already downloaded
def download_reuters_with_retries():
    retries = 3
    for _ in range(retries):
        try:
            nltk.download('reuters')
            nltk.download('stopwords')
            nltk.download('punkt')
            return True
        except Exception as e:
            st.error(f"Error downloading NLTK resources: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    return False

# Attempt to download the required data
if not download_reuters_with_retries():
    st.error("Failed to download necessary NLTK data after multiple attempts.")
else:
    # Cache the corpus to avoid re-downloading on each run
    @st.cache_data
    def load_reuters():
        try:
            documents = [reuters.raw(fileid) for fileid in reuters.fileids()]
            return documents
        except Exception as e:
            st.error(f"Error loading Reuters corpus: {e}")
            return []

    documents = load_reuters()

    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        return [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]

    tokenized_docs = [preprocess(doc) for doc in documents]

    # Train a Word2Vec model
    w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)

    # Compute document vectors
    def document_vector(doc):
        vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    doc_vectors = np.array([document_vector(doc) for doc in tokenized_docs])

    # Streamlit UI
    st.title('Semantic Search Engine')
    user_query = st.text_input('Enter your search query:')

    if st.button('Search') and user_query:
        query_tokens = preprocess(user_query)
        query_vector = document_vector(query_tokens).reshape(1, -1)
        similarities = cosine_similarity(query_vector, doc_vectors)[0]

        # Get top-k most relevant documents
        top_k = 5
        top_indices = similarities.argsort()[-top_k:][::-1]

        st.write('Top relevant search results:')
        for idx in top_indices:
            st.write(f"**Score:** {similarities[idx]:.4f}\n")
            st.write(documents[idx][:500] + '...')

