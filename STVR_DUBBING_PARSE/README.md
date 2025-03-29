# Screenplay Parser with LLM Agents

This application uses advanced LLM (Large Language Model) agents to intelligently parse and analyze screenplay documents with inconsistent formatting.

## Features

- Parses screenplays from TXT or DOCX files
- Identifies characters, locations, and dialogue
- Normalizes inconsistent character names and formatting
- Recognizes special audio notations (VO, MO, etc.)
- Provides summary statistics and visualizations
- Exports results as JSON or CSV

## Project Structure

```
screenplay-parser/
├── main.py                 # Main Streamlit application
├── config.py               # Configuration settings
├── file_utils.py           # File handling utilities
├── llm_agent.py            # Base LLM Agent class
├── agents.py               # Agent import module
├── segmentation_agent.py   # Document segmentation agent
├── entity_agent.py         # Entity recognition agent
├── dialogue_agent.py       # Dialogue processing agent
├── correction_agent.py     # Inconsistency correction agent
├── processor.py            # Main screenplay processor
├── __init__.py             # Package initialization
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/screenplay-parser.git
cd screenplay-parser
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

### LLM Providers

The app supports the following LLM providers:

1. **OpenAI** - Requires an API key
   - Models: GPT-3.5-Turbo, GPT-4, GPT-4-Turbo

2. **Ollama** - Requires a local Ollama instance
   - Models: Mistral, Llama3.1, Gemma3, etc.

3. **DeepSeek** - Uses DeepSeek models via Ollama
   - Models: DeepSeek Coder, DeepSeek r1

## How It Works

1. **Document Segmentation**: Breaks the screenplay into logical parts
2. **Entity Recognition**: Identifies characters, locations, and audio notations
3. **Dialogue Processing**: Normalizes dialogue and speaker information
4. **Correction**: Fixes inconsistencies across the document

## Advanced Configuration

You can adjust processing parameters in the sidebar:
- **Processing Granularity**: Control the chunk size for processing
- **Request Timeout**: Set maximum time for model requests
- **Detailed Progress**: Show detailed processing information
- **Debug Mode**: Display detailed debugging information

## Example Output

The application provides several tabs for analyzing the parsed screenplay:
- **Summary**: Key statistics and visualizations
- **Characters**: List of characters and dialogue counts
- **Scenes**: Breakdown of scenes with timestamps
- **Dialogue**: Sample dialogue with speaker information
- **Raw Data**: Export options for JSON and CSV

## License

MIT License