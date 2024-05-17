import streamlit as st
from nltk import ngrams
import nltk
nltk.download('punkt')

def extract_ngrams(text, n):
    tokens = nltk.word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    return n_grams

def main():
    st.title("N-gram Extractor")

    # User input
    text_input = st.text_area("Enter a text passage:", "")

    # Choose n for n-grams
    n = st.slider("Select the value of n for n-grams:", min_value=1, max_value=5, value=2)

    if st.button("Extract N-grams"):
        if not text_input:
            st.warning("Please enter a text passage.")
        else:
            # Extract and display n-grams
            n_grams_result = extract_ngrams(text_input, n)
            st.subheader(f"{n}-grams:")
            st.write(n_grams_result)

if __name__ == "__main__":
    main()