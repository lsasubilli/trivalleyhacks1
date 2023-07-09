import spacy
import pytextrank
import streamlit as st
from annotated_text import annotated_text
from pprint import pprint
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse


st.markdown('### YouTube / Google Drive Videos Summary')


tiinput = st.text_input('Enter Link', '')

st.text(tiinput)

example_text = tiinput

tx = YouTubeTranscriptApi.get_transcript('yBCAv_NzzPQ')


transcript=''
for value in tx:
    for key, val in value.items():
        if key=="text":
            transcript+=val

l = transcript.splitlines()
final_tra = " ".join(l)





nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

doc = nlp(final_tra)

st.subheader("Video Summary: ")

for sent in doc._.textrank.summary(limit_sentences=4):
   st.markdown(sent)




phrases_and_ranks = [(phrase.chunks[0], phrase.rank) for phrase in doc._.phrases]


sorted_phrases_and_ranks = sorted(phrases_and_ranks, key=lambda x: x[1], reverse=True)

top_10_topics = [str(item[0].text) for item in sorted_phrases_and_ranks[:10]]

# Join the topics with commas
topics_string = ', '.join(top_10_topics)
st.subheader('Top 10 Relevant Topics')
# Print the topics string
st.markdown(topics_string)

st.subheader('Relevant YouTube Videos')
base_url = 'https://www.youtube.com/results?search_query='

# Sort the phrases and ranks based on rank in descending order
sorted_phrases_and_ranks = sorted(phrases_and_ranks, key=lambda x: x[1], reverse=True)

# Extract the top 10 topics
top_10_topics = [str(item[0].text) for item in sorted_phrases_and_ranks[:15]]

# Create a Markdown string with all the links
links = []
for topic in top_10_topics:
    encoded_topic = urllib.parse.quote(topic)
    link = f"[{topic}]({base_url}{encoded_topic.replace(' ', '+')})"
    links.append(link)
links_string = '  \n'.join(links)

# Display the links in Streamlit
st.markdown(links_string, unsafe_allow_html=True)

st.success("Task Complete")