import streamlit as st
import numpy as np
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# Download necessary NLTK data
import nltk

# Download NLTK resources only if not already downloaded
try:
    nltk.data.find('corpora/reuters')
except LookupError:
    nltk.download('reuters')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
# Load and preprocess the Reuters dataset
try:
    documents = [reuters.raw(fileid) for fileid in reuters.fileids()]
except Exception as e:
    st.error(f"Error loading Reuters corpus: {e}")
    documents = []  # Ensuring an empty list is used if an error occurs


# Load and preprocess the Reuters dataset
documents = [reuters.raw(fileid) for fileid in reuters.fileids()]
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
