import streamlit as st
import spacy
from spacy import displacy
from newspaper import Article
import en_core_web_sm
nlp = en_core_web_sm.load()
from pprint import pprint

st.title("NER DEMO")

name = st.text_input("Enter URL")
st.write("OR")

text = st.text_area("Enter a paragraph")

if(st.button("ANALYSE")):

    if (text) :
        doc = nlp(text)
        ent_html = displacy.render(doc, style="ent", jupyter=False)
        # Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)
    elif(name):
        article = Article(name)
        article.download()
        article.parse()
        txt = article.text
        doc = nlp(txt)
        ent_html = displacy.render(doc, style="ent", jupyter=False)
        # Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)
    else:
        st.write("Enter URL or Paragraph")   