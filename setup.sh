#!/bin/bash

# Pull the Llama 3.1 8B model
ollama pull llama3.1

# Set up Streamlit config
mkdir -p ~/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

# Any other setup steps your app needs