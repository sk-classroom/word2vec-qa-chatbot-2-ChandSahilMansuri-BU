### Assignment guidance
# The goal of this assignment is to create a word2vec-based question-answer chatbot application that should be able
# to give the best answer based on vector search toward both question set and answer set. 
# Our exercise only showed how to apply question set for vector search. You can follow the hints to generate the chatbot. 
# What you need to submit for this assignment: an app url (You should publish your chatbot application on Streamlit Cloud. 
# Your chatbot assignment will be evaluated based on query questions listed as below:
# (1) A year before improving and popularizing the electrophorus, what did Volta become?
# (2) Does the Hymenoptera order include ants?
# (3) Who invented the voltaic pile?
# (4) Does Avogadro Law talk about the relationship between same volume masses?
###

import streamlit as st
import pandas as pd
import faiss
import gensim
import numpy as np

# load question-answer dataset 
df = pd.read_csv("data/Question_Answer_Dataset_v1.2_S10.csv")

# load question and answer vectors generated from pre-trained word2vec model
vector = ...
ques_vec = ...
ans_vec = ...

# load th trained word2vec model 
# Hint: You should use the word2vec model pre-trained with both question and answer sets.
trained_w2v = gensim.models.Word2Vec.load("data/w2v.model")

# App title
st.set_page_config(page_title="Word2vec Question and Answer Chatbot")

# Add header image 
st.image("data/header-chat-box.png")

# chat title 
st.title("Word2vec Question and Answer Chatbot")

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messagess
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate the embedding for query question
def trained_sentence_vec(sent):
    qu_voc = [word for word in sent if word in trained_w2v.wv.vocab]
    emb = [trained_w2v.wv[word] for word in qu_voc]
    ave_vec = np.mean(emb, axis=0)
    return ave_vec

# Function to find the answer through vector search
def find_answer(qr_sentence, ques_vec, ans_vec):
    qr_sentence = gensim.utils.simple_preprocess(qr_sentence)
    qr_sent_vec = trained_sentence_vec(qr_sentence)

    # Concatenate question and answer vectors
    all_vec = np.concatenate((ques_vec, ans_vec), axis=0)

    # Build the FAISS index
    index = faiss.index_factory(all_vec.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(all_vec)

    # Perform vector search
    faiss.normalize_L2(qr_sent_vec)
    similarity, idx = index.search(np.expand_dims(qr_sent_vec, axis=0), k=1)

    # Get the index of the optimal answer
    ans_idx = idx[0][0]

    return ans_idx


# User-provided prompt
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans_idx = find_answer(prompt, ques_vec)
            response = df["Answer"][ans_idx]
            st.write(response)
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

