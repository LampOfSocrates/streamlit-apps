#https://github.com/subratamondal1/EmotionAI/tree/main/emotionai
# 

import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title = "EmotionAI",
    page_icon = "😀",
    layout = "centered",
    initial_sidebar_state = "collapsed"
)

st.markdown(
    body = "# <center>EmotionAI: Sentiment Analysis with Fine-Tuned DistilBERT</center>",
    unsafe_allow_html = True
)
st.caption("EmotionAI is a powerful web application that leverages machine learning to perform real-time sentiment analysis on text data. At its core, EmotionAI utilizes a fine-tuned DistilBERT model, a lightweight variant of the popular BERT architecture, to accurately classify emotions conveyed in natural language inputs.")
st.caption("Trained on a custom dataset, the DistilBERT model achieved an impressive 92% accuracy in emotion classification, demonstrating its capability to understand and interpret the nuances of human emotions expressed through text. The model was fine-tuned using Hugging Face's transformers, datasets, and evaluate libraries, ensuring seamless integration with state-of-the-art natural language processing (NLP) tools.")
st.caption("The application's user-friendly interface, built with Streamlit, allows users to effortlessly input text and receive real-time emotion predictions. EmotionAI's intuitive design and responsive feedback make it an invaluable tool for sentiment analysis tasks, enabling users to gain valuable insights into the emotional tone of their textual data. Capable of distinguishing between six emotions: `Joy`, `Sadness`, `Anger`, `Fear`, `Love`, `Surprise`")

text_input = st.text_area(
    label = "Provide your text.",
    value = "I saw a movie today and it was really good."
)

model_id = "iamsubrata/distilbert-base-uncased-finetuned-emotion"

classifier = pipeline(
    task = "text-classification",
    model = model_id
)

if text_input:
    button  = st.button(
        label = "Emotion",
        key = "emotion_button"
    )

    if button:
        labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        predictions = classifier(
            text_input, 
            top_k = 6
        )
        predictions_df = pd.DataFrame(predictions)
        predictions_df.columns = ["Emotion", "Score"]
        predictions_df["Emotion"] = predictions_df["Emotion"].apply(lambda x: labels[int(x[-1])])
        predictions_df.sort_values(
            by = "Score",
            ascending = False
        )

        st.bar_chart(
            data = predictions_df,
            x = "Emotion",
            y = "Score"
        )