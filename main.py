import streamlit as st


from annotated_text import annotated_text

import transformers
from transformers import pipeline

from transformers import PegasusForConditionalGeneration, PegasusTokenizer


model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)



st.markdown('# Video Transcription Project')
my_list = [
    "Made",
    [
        " by:",
        ("Aadi", "Patangi"),
        " and ",
    ],
    ("Lalith", "Sasubilli"),
    ".",
]

annotated_text(my_list)
st.markdown('### This text-summarizing model was created using a refined version of the pre-trained Pegasus model,a state-of-the-art model for abstractive text summarization developed by the Google Research Team.')

tiinput = st.text_input('Text Input', '')

st.text(tiinput)

example_text = tiinput

tokens = pegasus_tokenizer(example_text, truncation=False  , padding="longest", return_tensors="pt")
print(tokens)
encoded_summary = pegasus_model.generate(**tokens)
print(encoded_summary)

decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)


summarizer = pipeline("summarization", model=model_name, tokenizer=pegasus_tokenizer, framework="pt")

summary = summarizer(example_text, min_length=0 , max_length=40)

# st.text(summary)
st.markdown(summary)

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# import transformers
# from transformers import PegasusTokenizer

# print( transformers.__version__)
