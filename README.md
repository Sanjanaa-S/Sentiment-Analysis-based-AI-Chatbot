# Sentiment-Analysis-based-AI-Chatbot

This repository contains the code for a simple AI chatbot powered by Microsoft DialoGPT and integrated with Sentiment Analysis. The chatbot is built using Streamlit, Transformers, and NLTK libraries, and is designed to interact with users in a conversational manner while also identifying the sentiment behind user input.

## Key Features:
- Dialog Generation: The chatbot uses the DialoGPT model (a conversational variant of GPT-2) to generate human-like responses to user input.
- Sentiment Analysis: The chatbot performs sentiment analysis on user input using the DistilBERT model, classifying the input as positive, negative, or neutral. This sentiment is then used to tailor the chatbot's responses.
- Real-Time Interaction: The chatbot allows real-time interactions, with the conversation history retained throughout the session, enabling a continuous and personalized chat experience.
- Customizable Response Prefixes: Based on the sentiment analysis, the chatbot alters its tone to provide context-sensitive responses (e.g., empathetic for negative sentiment or encouraging for positive sentiment).
