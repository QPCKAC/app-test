#!/bin/bash

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Install other dependencies
pip install -r requirements.txt

# Pull the Llama 3.1 8B model
ollama pull llama3.1

# Set up Streamlit config
mkdir -p ~/.streamlit/
echo "[server]
headless = true
enableCORS = false
" > ~/.streamlit/config.toml
