"""
Configuration settings for the Screenplay Parser App
"""
import streamlit as st

# Constants
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_OLLAMA_MODEL = "gemma3"
TEMPERATURE = 0.01
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def setup_sidebar_config():
    """Set up and return the configuration from the sidebar."""
    st.sidebar.title("Configuration")

    # Provider selection dropdown
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["OpenAI", "Ollama", "DeepSeek"],  # Provider options
        key="llm_provider_select"
    )

    # Initialize configuration dictionary
    config = {
        "llm_provider": llm_provider,
        "api_key": None,
        "model": None,
        "ollama_url": None,
        "use_code_format": False,
        "parsing_granularity": 1000,
        "timeout_seconds": 30,
        "detailed_progress": True,
        "debug_mode": False
    }

    # Provider-specific configurations
    if llm_provider == "OpenAI":
        config["api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
        config["model"] = st.sidebar.selectbox(
            "Model", 
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            key="openai_model_select"
        )
        
    elif llm_provider == "DeepSeek":  # Special handling for DeepSeek models
        config["ollama_url"] = st.sidebar.text_input("Ollama URL", value="http://localhost:11434", key="deepseek_url")
        config["model"] = st.sidebar.selectbox(
            "Model", 
            ["deepseek-coder:6.7b", "deepseek-coder:latest", "deepseek-r1:8b"],
            index=0,
            key="deepseek_model_select"
        )
        
        # Add specific DeepSeek options
        st.sidebar.write("DeepSeek Options:")
        config["use_code_format"] = st.sidebar.checkbox("Use code-optimized prompts", value=True, key="use_code_format")
        st.sidebar.info("DeepSeek models work best with code-optimized prompts for structured data.")
        
    else:  # Ollama
        config["ollama_url"] = st.sidebar.text_input("Ollama URL", value="http://localhost:11434", key="ollama_url")
        config["model"] = st.sidebar.selectbox(
            "Model", 
            ["mistral", "llama3.1", "nomic-embed-text", "gemma3"],
            index=0,
            key="ollama_model_select"
        )

    # Common processing options
    config["parsing_granularity"] = st.sidebar.slider(
        "Processing Granularity",
        min_value=1000,
        max_value=5000,
        value=2000,
        step=1000,
        help="Number of characters to process at once. Lower values are slower but more reliable.",
        key="parsing_granularity"
    )

    config["timeout_seconds"] = st.sidebar.slider(
        "Request Timeout (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30,
        help="Maximum time to wait for each model request. Increase for complex chunks.",
        key="timeout_seconds"
    )

    config["detailed_progress"] = st.sidebar.checkbox(
        "Show detailed progress", 
        value=True,
        key="detailed_progress",
        help="Show more detailed progress information during processing"
    )

    config["debug_mode"] = st.sidebar.checkbox("Debug Mode", value=False, key="debug_mode")

    return config