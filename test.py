import streamlit as st
from annotated_text import annotated_text

import transformers
from transformers import pipeline
from gtts import gTTS
from google_trans_new import google_translator 


from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from pprint import pprint
from youtube_transcript_api import YouTubeTranscriptApi

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
st.markdown('### This text-summarizing model was trained on a 10.8 GB Dataset from Kaggle, a state-of-the-art model for abstractive text summarization developed by us.')

tiinput = st.text_input('Text Input', '')

st.text(tiinput)

example_text = tiinput



tokens = pegasus_tokenizer(example_text, truncation=False  , padding="longest", return_tensors="pt")
encoded_summary = pegasus_model.generate(**tokens)

tone = "neutral"
decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)


summarizer = pipeline("summarization", model=model_name, tokenizer=pegasus_tokenizer, framework="pt")

summary = summarizer(example_text, min_length=30 , max_length=150)
text = "Summary: " + summary[0]['summary_text']
st.write(f"Translate your thoughts.") 
input_text = st.text_input('Enter whatever')
if st.button('Translate'):    
   result = trans.translate(input_text, lang_tgt = 'ja')
   st.success(result)             
   speech = gTTS(text = result, lang = 'ja', slow = False)




st.balloons()
# st.text(summary)
st.markdown(text)
st.success("Summary Generated.")




