#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting environment setup for MSME Invoice Agent..."

# 1. Initialize Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "✅ Python virtual environment already exists."
fi

# Activate virtual environment
source venv/Scripts/activate

# 2. Upgrade pip and install all Python dependencies
echo "📥 Installing required Python packages..."
pip install --upgrade pip
pip install sqlalchemy psycopg2-binary pgvector pypdf langchain langchain-ollama langchain-core langchain-classic langchain-text-splitters streamlit fastapi uvicorn

# 3. Check for Ollama installation
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed on this system."
    echo "👉 Please download and install it manually from: https://ollama.com"
    echo "👉 After installing, re-run this script to pull the required AI models."
    exit 1
else
    echo "✅ Ollama installation detected."
fi

# 4. Boot up Ollama models locally
echo "🤖 Pulling local LLM (llama3.2)..."
ollama pull llama3.2

echo "🧠 Pulling local Vector Embedding Model (nomic-embed-text)..."
ollama pull nomic-embed-text

echo "🎉 Setup complete! To activate your environment, run: source venv/Scripts/activate"
