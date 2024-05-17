import streamlit as st
from transformers import pipeline

# Load the NER model and tokenizer from Hugging Face
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

model = load_model()

# Streamlit app
st.title("Named Entity Recognition with BERT")
st.write("Enter a sentence to see the named entities recognized by the model.")

# Text input
text = st.text_area("Enter your sentence here:")

# Perform NER and display results
if text:
    st.write("Entities recognized:")
    entities = model(text)
   
    # Create a dictionary to map entity labels to colors
    label_colors = {
        'ORG': 'lightblue',
        'PER': 'lightgreen',
        'LOC': 'lightcoral',
        'MISC': 'lightyellow'
    }
   
    # Prepare the HTML output with styled entities
    def get_entity_html(text, entities):
        html = ""
        last_idx = 0
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['entity']
            entity_text = text[start:end]
            color = label_colors.get(label, 'lightgray')
           
            # Append the text before the entity
            html += text[last_idx:start]
            # Append the entity with styling
            html += f'<mark style="background-color: {color}; border-radius: 3px;">{entity_text}</mark>'
            last_idx = end
       
        # Append any remaining text after the last entity
        html += text[last_idx:]
        return html
   
    # Generate and display the styled HTML
    styled_text = get_entity_html(text, entities)
    st.markdown(styled_text, unsafe_allow_html=True)
