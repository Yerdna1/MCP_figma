#!/bin/bash
# Setup script for Ollama local LLM installation

# Check if Ollama is already installed
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama is already installed."
else
    echo "Installing Ollama..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        brew install ollama
    else
        echo "Unsupported OS. Please visit https://ollama.com for installation instructions."
        exit 1
    fi
fi

# Start Ollama if not running
if ! pgrep -x "ollama" >/dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5  # Wait for service to start
fi

# Pull recommended models
echo "Pulling recommended models for screenplay parsing..."
ollama pull llama2
echo "Pulling mistral model (recommended for best performance)..."
ollama pull mistral

echo "Ollama setup complete! The service is running at http://localhost:11434"
echo "You can now run the screenplay parser app with: streamlit run app.py"