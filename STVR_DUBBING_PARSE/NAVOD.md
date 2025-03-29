Here's a complete Python and Streamlit application for parsing screenplays using LLM agents. This solution addresses the challenges of inconsistent formatting, typos, and varied notation styles in screenplay documents.
Key Components of the Solution

Complete Streamlit App (screenplay-parser-app.py):

A full-featured web application that provides an intuitive interface for screenplay processing
Supports uploading Word (.docx) or text (.txt) files
Visualizes the parsed screenplay with character analysis, scene breakdown, and dialogue insights


Multi-Agent LLM Architecture:

Document Segmentation Agent: Splits the screenplay into logical components
Entity Recognition Agent: Identifies characters, locations, and notation styles
Dialogue Processing Agent: Normalizes character names and dialogue formatting
Correction Agent: Fixes inconsistencies across the document


Flexible LLM Provider Options:

OpenAI integration (GPT-3.5, GPT-4) for high-accuracy parsing
Local Ollama support for privacy and cost-efficient processing
Configurable processing parameters


Setup and Deployment Files:

requirements.txt: Lists all Python dependencies
setup_ollama.sh: Script to set up local LLM processing
Dockerfile and docker-compose.yml: For containerized deployment



How to Use This Solution

Installation:
bashCopypip install -r requirements.txt

Launch the App:
bashCopystreamlit run screenplay-parser-app.py

Configuration:

Choose your LLM provider (OpenAI or local Ollama)
Set your API key if using OpenAI
Adjust processing parameters as needed


Processing:

Upload your screenplay document
Click "Process Screenplay"
Explore the analysis across different tabs



For local LLM processing with Ollama, run the included setup script or use Docker Compose to launch both the app and Ollama service together.
The intelligent agents will work together to identify screenplay elements even with inconsistent formatting, making this solution much more robust than traditional pattern-based parsers.