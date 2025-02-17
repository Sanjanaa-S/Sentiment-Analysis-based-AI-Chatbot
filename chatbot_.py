import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

# Initialize models
st.title("💬 AI Chatbot")
st.write("Powered by Microsoft DialoGPT & Sentiment Analysis")

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Constants
MAX_HISTORY_LENGTH = 1000
MIN_RESPONSE_LENGTH = 10

# Functions
def preprocess(text):
    tokens = word_tokenize(text.lower())
    lem = WordNetLemmatizer()
    lemmatized_tokens = [lem.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def sentiment_analysis(text):
    result = sentiment_pipeline(text)
    sentiment = result[0]['label'].lower()
    return sentiment, result[0]['score']

def generate_response(prompt, conversation_history):
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    if conversation_history is not None:
        bot_input_ids = torch.cat([conversation_history, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    if bot_input_ids.shape[-1] > MAX_HISTORY_LENGTH:
        bot_input_ids = bot_input_ids[:, -MAX_HISTORY_LENGTH:]

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=MAX_HISTORY_LENGTH,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        min_length=MIN_RESPONSE_LENGTH
    )
    
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip(), chat_history_ids

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input UI
user_input = st.text_input("Type your message:", "", key="user_input")
send_button = st.button("Send")

if send_button and user_input:
    cleaned_input = preprocess(user_input)
    sentiment, confidence = sentiment_analysis(cleaned_input)
    
    # Sentiment-based response prefixes
    sentiment_prefix = {
        "positive": "I'm glad you're feeling good! ",
        "negative": "I understand you're feeling down. ",
        "neutral": "I see. "
    }
    
    prompt = sentiment_prefix.get(sentiment, "") + cleaned_input
    response, new_history = generate_response(prompt, st.session_state.chat_history)

    # Update session state
    st.session_state.chat_history = new_history
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

# Display chat history
st.subheader("🗨️ Chat History")
for sender, msg in st.session_state.messages:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(f"**{sender}:** {msg}")

# Reset button
if st.button("Reset Chat"):
    st.session_state.chat_history = None
    st.session_state.messages = []
    st.experimental_rerun()
