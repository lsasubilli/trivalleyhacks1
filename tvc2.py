#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install Transformers==3.2.0')


# In[2]:


get_ipython().system('pip3 install SentencePiece')


# In[3]:


from transformers import PegasusForConditionalGeneration, PegasusTokenizer


# In[5]:


model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)


# In[27]:


example_text = """The United States is by far the most successful country in Olympic basketball, with United States men's teams having won 16 of 19 tournaments in which they participated, including seven consecutive titles from 1936 through 1968. United States women's teams have won eight titles out of the 10 tournaments in which they competed, including seven in a row from 1996 to 2020. Besides the United States, Argentina is the only nation still in existence which has won either the men's or women's tournament. The Soviet Union, Yugoslavia and the Unified Team are the countries no longer in existence who have won the tournament. The United States are the defending champions in both men's and women's tournaments."""

tokens = pegasus_tokenizer(example_text, truncation=False  , padding="longest", return_tensors="pt")

tokens


# In[28]:


encoded_summary = pegasus_model.generate(**tokens)


# In[29]:


decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)


# In[30]:


print(decoded_summary)


# In[ ]:


get_ipython().system('pip3 install pipeline')


# In[ ]:


from transformers import pipeline

summarizer = pipeline("summarization", model=model_name, tokenizer=pegasus_tokenizer, framework="pt")


# In[ ]:


summary = summarizer(example_text, min_length=30, max_length=150)


# In[36]:


summary[0]["summary_text"]


# In[35]:





# In[ ]:




