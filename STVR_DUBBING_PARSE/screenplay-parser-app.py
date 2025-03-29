"""
Screenplay Parser App with LLM Agents
This app uses LLM agents to intelligently parse screenplay documents with inconsistent formatting.
"""
import streamlit as st
import os
import json
import re
import time
import pandas as pd
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
import docx2txt
import requests
from datetime import datetime

# Constants
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_OLLAMA_MODEL = "deepseek-coder"
TEMPERATURE = 0.01
MAX_RETRIES = 3
RETRY_DELAY = 4  # seconds

# Configure page
# Sidebar with configuration options
st.sidebar.title("Configuration")

# First, let's modify the provider selection dropdown
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["OpenAI", "Ollama", "DeepSeek"],  # Provider options
    key="llm_provider_select"
)

if llm_provider == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key")
    model = st.sidebar.selectbox("Model", 
                               ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                               index=0,
                               key="openai_model_select")
elif llm_provider == "DeepSeek":  # Special handling for DeepSeek models
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434", key="deepseek_url")
    model = st.sidebar.selectbox("Model", 
                           ["deepseek-coder:6.7b", "deepseek-coder:latest","deepseek-r1:8b"],
                           index=0,
                           key="deepseek_model_select")
    
    # Add specific DeepSeek options
    st.sidebar.write("DeepSeek Options:")
    use_code_format = st.sidebar.checkbox("Use code-optimized prompts", value=True, key="use_code_format")
    st.sidebar.info("DeepSeek models work best with code-optimized prompts for structured data.")
else:  # Ollama
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434", key="ollama_url")
    model = st.sidebar.selectbox("Model", 
                           ["mistral", "llama3.1", "nomic-embed-text", "gemma3"],
                           index=0,
                           key="ollama_model_select")  # Added gemma3 to Ollama models

# Define processing options with more conservative limits
parsing_granularity = st.sidebar.slider(
    "Processing Granularity",
    min_value=200,
    max_value=1000,
    value=300,
    step=100,
    help="Number of characters to process at once. Lower values are slower but more reliable.",
    key="parsing_granularity"
)

# Add timeout control
timeout_seconds = st.sidebar.slider(
    "Request Timeout (seconds)",
    min_value=30,
    max_value=300,
    value=120,
    step=30,
    help="Maximum time to wait for each model request. Increase for complex chunks.",
    key="timeout_seconds"
)

# Add progress reporting option
detailed_progress = st.sidebar.checkbox("Show detailed progress", 
                                      value=True,
                                      key="detailed_progress",
                                      help="Show more detailed progress information during processing")

debug_mode = st.sidebar.checkbox("Debug Mode", value=False, key="debug_mode")

# Helper functions for file handling
def read_file(uploaded_file) -> str:
    """Read content from an uploaded file."""
    if uploaded_file.name.endswith('.docx'):
        # Create a temporary file to save the uploaded docx
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Extract text from the docx file
        text = docx2txt.process(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        return text
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.getvalue().decode('utf-8')
    else:
        st.error("Unsupported file format. Please upload a .txt or .docx file.")
        return ""

# LLM Agent classes
class LLMAgent:
    """Base class for LLM agents"""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, ollama_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url
        
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with the given prompt and return the response."""
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "OpenAI":
                    return self._call_openai(prompt, system_prompt)
                elif self.provider == "DeepSeek" or (self.provider == "Ollama" and "deepseek" in self.model.lower()):
                    # Use special DeepSeek handling
                    return self._call_deepseek(prompt, system_prompt)
                elif self.provider == "Ollama":
                    return self._call_ollama(prompt, system_prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    if detailed_progress:
                        st.warning(f"Retrying after error: {str(e)}")
                    continue
                else:
                    st.error(f"Error calling LLM: {str(e)}")
                    return ""

    def _call_deepseek(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with optimized handling for DeepSeek Coder models."""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            # DeepSeek Coder prompt optimization - function format works better
            code_prompt = f"""// Function to parse screenplay text
    function parseScreenplaySegment(text) {{
    // Input text to parse:
    /*
    {prompt.replace('```', '').strip()}
    */

    // Return a JSON array with parsed elements (scene headers, dialogue, etc.)
    // IMPORTANT: The response MUST be a valid, complete JSON array
    return [
        // Example format - to be replaced with actual parsed content:
        {{
        "type": "scene_header",
        "scene_type": "INT",
        "timecode": "00:01:23"
        }},
        {{
        "type": "dialogue",
        "character": "CHARACTER_NAME",
        "audio_type": "VO",
        "text": "Dialogue text"
        }}
    ];
    }}

    // Call the function and return ONLY the JSON result:
    parseScreenplaySegment();
    """

            # Use code-optimized format if selected, otherwise use standard format
            if hasattr(self, 'use_code_format') and self.use_code_format:
                full_prompt = code_prompt
            else:
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nRespond with JSON only."
                else:
                    full_prompt = f"User: {prompt}\n\nRespond with JSON only."
            
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,       # Very low temperature for structured output
                "num_predict": 12000,       # Equivalent to max_tokens in OpenAI
                "stop": ["```", "```json"], # Stop sequences to avoid markdown wrapping
                "stream": False            # Ensure we get a complete response at once
            }
            
            if detailed_progress:
                st.write(f"Calling DeepSeek model: {self.model}")
            
            if debug_mode:
                st.write(f"DeepSeek prompt length: {len(full_prompt)} characters")
            
            # Make the API call with configured timeout
            response = requests.post(url, json=data, timeout=timeout_seconds)
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Attempt to complete the JSON if it appears to be truncated
                if raw_response.count("[") > raw_response.count("]") or raw_response.count("{") > raw_response.count("}"):
                    raw_response = self._complete_json(raw_response)
                    
                # Aggressive JSON extraction specifically for DeepSeek
                # First try to find array pattern
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_response, re.DOTALL)
                if json_match:
                    raw_response = json_match.group(0)
                else:
                    # If no array found, try to find any valid JSON object
                    obj_match = re.search(r'\{\s*"[^"]+"\s*:.*?\}', raw_response, re.DOTALL)
                    if obj_match:
                        raw_response = f"[{obj_match.group(0)}]"
                
                # Clean the response
                cleaned_response = self._clean_response(raw_response)
                
                # Log the raw and cleaned responses if debug mode is enabled
                if debug_mode:
                    st.write("Raw DeepSeek response:")
                    st.text(raw_response[:500] + ("..." if len(raw_response) > 500 else ""))
                    st.write("Cleaned response:")
                    st.text(cleaned_response[:500] + ("..." if len(cleaned_response) > 500 else ""))
                    
                return cleaned_response
            else:
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
            return "[]"  # Return empty JSON array on connection error
        except requests.exceptions.Timeout:
            st.warning(f"DeepSeek request timed out after {timeout_seconds} seconds.")
            return "[]"  # Return empty JSON array on timeout
        except Exception as e:
            st.error(f"Error calling DeepSeek: {str(e)}")
            return "[]"  # Return empty JSON array on any error

    def _complete_json(self, partial_json: str) -> str:
        """Attempt to complete truncated JSON responses."""
        # Count brackets to detect incomplete JSON
        open_brackets = partial_json.count("[")
        close_brackets = partial_json.count("]")
        open_braces = partial_json.count("{")
        close_braces = partial_json.count("}")
        
        # First, ensure we have a starting bracket
        if "[" not in partial_json and open_braces > 0:
            partial_json = "[" + partial_json
            open_brackets += 1
        
        # Add missing closing braces
        if open_braces > close_braces:
            partial_json += "}" * (open_braces - close_braces)
        
        # Add missing closing brackets
        if open_brackets > close_brackets:
            partial_json += "]" * (open_brackets - close_brackets)
        
        return partial_json
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API."""
        import openai
        
        openai.api_key = self.api_key
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        return response.choices[0].message.content
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with improved response handling."""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            # Format the prompt properly
            if system_prompt:
                # Use a clear separator between system prompt and user prompt
                full_prompt = f"""SYSTEM INSTRUCTION:
    {system_prompt}

    USER:
    {prompt}

    IMPORTANT: Respond with valid, parseable JSON only. No explanations or other text.
    Example format:
    [
    {{
        "type": "scene_header",
        "scene_type": "INT",
        "timecode": "00:01:23"
    }},
    {{
        "type": "dialogue",
        "character": "CHARACTER_NAME",
        "audio_type": "VO",
        "text": "Dialogue text"
    }}
    ]
    """
            else:
                full_prompt = f"""USER:
    {prompt}

    IMPORTANT: Respond with valid, parseable JSON only. No explanations or other text.
    """
                
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "temperature": 0.01,  # Lower temperature for more predictable JSON
                "stream": False       # Ensure we get a complete response at once
            }
            
            st.write(f"Calling Ollama model: {self.model}")
            if debug_mode:
                st.write(f"Prompt length: {len(full_prompt)} characters")
            
            # Make the API call with extended timeout
            response = requests.post(url, json=data, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                
                # Try to extract JSON array more aggressively
                json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_response, re.DOTALL)
                if json_match:
                    raw_response = json_match.group(0)
                
                # Clean the response
                cleaned_response = self._clean_response(raw_response)
                
                # Log the raw and cleaned responses if debug mode is enabled
                if debug_mode:
                    st.write("Raw Ollama response:")
                    st.text(raw_response[:1000] + ("..." if len(raw_response) > 1000 else ""))
                    st.write("Cleaned response:")
                    st.text(cleaned_response[:1000] + ("..." if len(cleaned_response) > 1000 else ""))
                    
                return cleaned_response
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
            return ""
        except Exception as e:
            st.error(f"Error calling Ollama: {str(e)}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Clean the LLM response to extract useful content."""
        # Remove style tags or any XML/HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove markdown code block syntax
        response = re.sub(r'```json|```python|```|~~~', '', response)
        
        # Remove any "JSON:" or similar prefixes
        response = re.sub(r'^.*?(\[|\{)', r'\1', response, flags=re.DOTALL)
        
        # If we have text after the JSON, remove it
        json_end_match = re.search(r'(\]|\})[^\]\}]*$', response)
        if json_end_match:
            end_pos = json_end_match.start() + 1
            response = response[:end_pos]
        
        # If response starts with backticks, remove them
        response = response.strip('`')
        
        # Fix common JSON errors
        response = response.replace("'", '"')  # Replace single quotes with double quotes
        response = re.sub(r',\s*(\}|\])', r'\1', response)  # Remove trailing commas
        
        return response.strip()

class DocumentSegmentationAgent(LLMAgent):
    """Agent for segmenting the document into logical parts."""
    
    # Modify your segment_document method
    def segment_document(self, text: str, chunk_size: int = 500) -> List[Dict]:
        """Segment document by logical patterns like scene markers."""
        # Try to identify scene markers or logical breaks
        scene_markers = re.findall(r'\*\*[A-Z]+\.\*\* --', text)
        
        if scene_markers and len(scene_markers) > 5:
            # If we have scene markers, split by those
            chunks = re.split(r'(\*\*[A-Z]+\.\*\* --)', text)
            processed_chunks = []
            
            # Recombine the splits to keep the markers with their content
            for i in range(1, len(chunks), 2):
                if i < len(chunks) - 1:
                    processed_chunks.append(chunks[i] + chunks[i+1])
        else:
            # Fall back to character-based chunking but with smaller size
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            processed_chunks = chunks
        
        all_segments = []
        for i, chunk in enumerate(processed_chunks):
            st.write(f"Processing chunk {i+1}/{len(processed_chunks)}...")
            segments = self._process_chunk(chunk, i)
            all_segments.extend(segments)
        
        return all_segments
    
    def _process_chunk(self, text: str, chunk_index: int) -> List[Dict]:
        """Process a single chunk of text with robust JSON parsing."""
        
        # Define the JSON schema for the updated format
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timecode": {"type": "string"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"}
                }
            }
        }
        
        system_prompt = f"""
        You are an expert screenplay parser. Your task is to analyze a screenplay segment and convert it into structured data.
        
        For each line or segment, extract these components (all are optional):
        1. TIMECODE - Any timestamps in the format like "00:05:44", "1:15:35", "**00:58\\-\\-\\-\\-\\-\\-\\-\\-\\--**", etc.
        2. SPEAKER - Names in ALL CAPS, which may include:
        - Multiple speakers separated by commas (e.g., "PETER, KAROL")
        - Speakers with numbers (e.g., "KAROL1")
        - Audio notation in parentheses like "(VO)", "(MO)", "(zMO)", etc.
        3. TEXT - The actual dialogue or action text
        
        The response MUST follow this exact JSON schema:
        {json.dumps(schema, indent=2)}
        
        IMPORTANT PARSING RULES:
        - If a line has no clear speaker, put any text content in the "text" field
        - Scene headers, stage directions, and other non-dialogue text should be included in "text"
        - Keep ALL speakers in uppercase as they appear in the original
        - Audio notations like "(VO)" or "(MO)" should be included as part of the speaker field
        - The first 1-2 pages may contain intro content in different formats - still parse them
        
        Examples of valid entries:
        [
        {{"timecode": "00:05:44", "speaker": "FERNANDO (MO)", "text": "Vstúpte."}},
        {{"speaker": "CABRERA (MO)", "text": "Veličenstvo..vaša manželka stanovila v poslednej vôli, aby boli uhradené jej dlhy."}},
        {{"timecode": "**06:12\\-\\-\\-\\-\\-\\-\\-\\-\\--**", "text": ""}},
        {{"speaker": "FUENSALIDA, CHACÓN (VO)", "text": "Avšak..list z Flámska sa zdržal."}},
        {{"text": "**INT.** -- 00:07:31"}}
        ]
        
        IMPORTANT: Return ONLY valid JSON and nothing else. No explanation, no markdown, just the JSON array.
        """
        
        prompt = f"""
        Analyze this screenplay segment and break it into structured data with timecode, speaker, and text fields.
        This is chunk {chunk_index} of a longer document.
        
        TEXT:
        ```
        {text}
        ```
        
        Return ONLY a JSON array following the specified schema. No explanations, no other text, just the JSON array.
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        # Try several strategies to extract valid JSON
        try:
            # Strategy 1: Try direct JSON loading
            segments = json.loads(response)
            return self._normalize_segments(segments)  # Apply normalization to handle inconsistencies
        except json.JSONDecodeError:
            st.warning(f"Direct JSON parsing failed. Trying to extract JSON from response...")
            
            # Strategy 2: Look for array pattern
            try:
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    segments = json.loads(json_str)
                    return self._normalize_segments(segments)  # Apply normalization
            except:
                pass
            
            # Strategy 3: Try to find and fix common JSON issues
            try:
                # Replace any markdown code block syntax
                cleaned_response = re.sub(r'```json|```|<.*?>|\n\s*\n', '', response)
                # Ensure the response starts with [ and ends with ]
                if not cleaned_response.strip().startswith('['):
                    cleaned_response = '[' + cleaned_response.strip()
                if not cleaned_response.strip().endswith(']'):
                    cleaned_response = cleaned_response.strip() + ']'
                    
                segments = json.loads(cleaned_response)
                return self._normalize_segments(segments)  # Apply normalization
            except:
                pass
            
            # Strategy 4: Fall back to a simpler prompt
            st.error("JSON extraction failed. Trying with a simpler prompt...")
            
            simpler_prompt = f"""
            Parse this screenplay segment into a simple JSON array of objects with three fields:
            - "timecode" (optional): Any timestamps
            - "speaker" (optional): Names in ALL CAPS (can include multiple speakers)
            - "text" (required): The content or dialogue
            
            TEXT:
            ```
            {text[:1000]}  # Shorten the text for the retry
            ```
            
            Return ONLY valid JSON, no other text.
            """
            
            try:
                simpler_response = self._call_llm(simpler_prompt, None)
                # Try direct JSON loading
                segments = json.loads(simpler_response)
                return self._normalize_segments(segments)  # Apply normalization
            except:
                # Final fallback: return a minimal structure
                st.error("All parsing attempts failed. Returning minimal structure.")
                # Create a single segment with the entire text
                return [{"text": text}]

    def _normalize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Normalize fields in segments to ensure consistency."""
        normalized = []
        for segment in segments:
            # Handle inconsistent field names
            if "characters" in segment and "speaker" not in segment:
                segment["speaker"] = segment.pop("characters")
            
            if "character" in segment and "speaker" not in segment:
                segment["speaker"] = segment.pop("character")
                
            if "audio_type" in segment and "speaker" in segment:
                # Add audio type to speaker if not already included
                if "(" not in segment["speaker"]:
                    segment["speaker"] = f"{segment['speaker']} ({segment.pop('audio_type')})"
                else:
                    # Remove audio_type as it's already in the speaker
                    segment.pop("audio_type")
                    
            # Ensure type info goes into text field
            if "type" in segment and segment["type"] not in ["dialogue", "speaker"]:
                if "text" not in segment or not segment["text"]:
                    segment["text"] = f"Type: {segment.pop('type')}"
                elif "scene_type" in segment or "timecode" in segment:
                    type_info = segment.pop("type")
                    scene_type = segment.pop("scene_type", "")
                    time_info = segment.pop("timecode", "")
                    segment["text"] = f"{type_info.upper()} {scene_type} {time_info} {segment.get('text', '')}"
            
            # Remove any old type field
            if "type" in segment:
                segment.pop("type")
            
            # Ensure all segments have a text field at minimum
            if "text" not in segment and "speaker" not in segment and "timecode" not in segment:
                continue  # Skip completely empty segments
            
            if "text" not in segment:
                segment["text"] = ""
            
            normalized.append(segment)
        
        return normalized

class EntityRecognitionAgent(LLMAgent):
    """Agent for identifying and normalizing entities in the screenplay."""
    
    def identify_entities(self, segments: List[Dict]) -> Dict:
        """Identify characters, locations, and other entities in the segments."""
        system_prompt = """
        You are an expert screenplay analyzer. Your task is to extract and normalize all entities from a screenplay.
        Focus on:
        1. Characters - Find all character names and normalize any inconsistencies
        2. Locations - Extract all locations mentioned
        3. Audio notations - Identify all audio notation types (MO, VO, zMO, etc.) and explain their meaning
        
        Format your response as JSON with these keys:
        - "characters": List of unique character names
        - "locations": List of unique locations
        - "audio_notations": Dictionary mapping audio notation to its meaning
        """
        
        # Extract dialogue segments to analyze characters
        dialogue_segments = [s for s in segments if s.get("type") == "dialogue"]
        
        # Extract scene headers to analyze locations
        scene_segments = [s for s in segments if s.get("type") == "scene_header"]
        
        prompt = f"""
        Analyze these screenplay segments and identify all entities.
        
        DIALOGUE SEGMENTS:
        ```
        {json.dumps(dialogue_segments, indent=2)}
        ```
        
        SCENE SEGMENTS:
        ```
        {json.dumps(scene_segments, indent=2)}
        ```
        
        Return ONLY the JSON without any additional explanation.
        """
        
        response = self._call_llm(prompt, system_prompt)
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
            else:
                entities = json.loads(response)
                
            return entities
        except json.JSONDecodeError:
            st.error(f"Failed to parse entity recognition response as JSON: {response}")
            return {
                "characters": [],
                "locations": [],
                "audio_notations": {}
            }


class DialogueProcessingAgent(LLMAgent):
    """Agent for understanding and normalizing dialogue."""
    
    def process_dialogue(self, dialogue_segments: List[Dict]) -> List[Dict]:
        """Process dialogue segments to normalize and fill in missing information."""
        if not dialogue_segments:
            return []
            
        system_prompt = """
        You are an expert dialogue analyzer for screenplays. Your task is to process dialogue segments,
        ensuring consistency in character names and audio notations.
        
        For each dialogue segment:
        1. Normalize character names to their canonical form
        2. Identify the correct audio notation type
        3. Clean the dialogue text, preserving stage directions within dialogue
        
        Return the processed dialogues in the same JSON format, but with normalized values.
        """
        
        # Process in batches to avoid hitting token limits
        batch_size = 20
        all_processed = []
        
        for i in range(0, len(dialogue_segments), batch_size):
            batch = dialogue_segments[i:i+batch_size]
            st.write(f"Processing dialogue batch {i//batch_size + 1}/{(len(dialogue_segments)-1)//batch_size + 1}...")
            
            prompt = f"""
            Process these dialogue segments from a screenplay to normalize character names,
            audio notations, and clean up dialogue text.
            
            ```
            {json.dumps(batch, indent=2)}
            ```
            
            Return ONLY the JSON array of processed dialogue segments.
            """
            
            response = self._call_llm(prompt, system_prompt)
            
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    processed_batch = json.loads(json_str)
                else:
                    processed_batch = json.loads(response)
                    
                all_processed.extend(processed_batch)
            except json.JSONDecodeError:
                st.error(f"Failed to parse dialogue processing response as JSON: {response}")
                all_processed.extend(batch)  # Use original batch on error
                
        return all_processed


class CorrectionAgent(LLMAgent):
    """Agent for identifying and correcting inconsistencies in the screenplay."""
    
    def correct_inconsistencies(self, segments: List[Dict], entities: Dict) -> List[Dict]:
        """Identify and correct inconsistencies in the segments based on entity knowledge."""
        system_prompt = """
        You are an expert screenplay editor. Your task is to identify and correct inconsistencies in a screenplay,
        including typos, formatting errors, and character name variations.
        
        Use the provided entity information to normalize:
        1. Character names - Ensure all references to the same character use a consistent name
        2. Audio notations - Standardize all audio notations to a consistent format
        3. Scene formatting - Ensure scene headers follow a standard format
        
        Return the corrected segments in the same JSON format, but with inconsistencies fixed.
        """
        
        # Process in batches to avoid hitting token limits
        batch_size = 30
        all_corrected = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            st.write(f"Correcting inconsistencies batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}...")
            
            prompt = f"""
            Correct inconsistencies in these screenplay segments based on the entity information provided.
            
            ENTITIES:
            ```
            {json.dumps(entities, indent=2)}
            ```
            
            SEGMENTS:
            ```
            {json.dumps(batch, indent=2)}
            ```
            
            Return ONLY the JSON array of corrected segments.
            """
            
            response = self._call_llm(prompt, system_prompt)
            
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    corrected_batch = json.loads(json_str)
                else:
                    corrected_batch = json.loads(response)
                    
                all_corrected.extend(corrected_batch)
            except json.JSONDecodeError:
                st.error(f"Failed to parse correction response as JSON: {response}")
                all_corrected.extend(batch)  # Use original batch on error
                
        return all_corrected


class ScreenplayProcessor:
    """Main processor that orchestrates the agents to parse and analyze a screenplay."""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, ollama_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url
        
        # Initialize agents
        self.segmentation_agent = DocumentSegmentationAgent(provider, model, api_key, ollama_url)
        self.entity_agent = EntityRecognitionAgent(provider, model, api_key, ollama_url)
        self.dialogue_agent = DialogueProcessingAgent(provider, model, api_key, ollama_url)
        self.correction_agent = CorrectionAgent(provider, model, api_key, ollama_url)
        
    def process_screenplay(self, text: str, chunk_size: int = 4000) -> Dict:
        """Process a screenplay through the full pipeline."""
        with st.status("Processing screenplay...", expanded=True) as status:
            # Step 1: Segment the document
            status.update(label="Segmenting document...")
            segments = self.segmentation_agent.segment_document(text, chunk_size)
            
            if not segments:
                st.error("Failed to segment document. Please try again.")
                return {"segments": [], "entities": {}}
            
            # Step 2: Identify entities
            status.update(label="Identifying entities...")
            entities = self.entity_agent.identify_entities(segments)
            
            # Step 3: Process dialogue specifically
            status.update(label="Processing dialogue...")
            dialogue_segments = [s for s in segments if s.get("type") == "dialogue"]
            processed_dialogue = self.dialogue_agent.process_dialogue(dialogue_segments)
            
            # Update the original segments with processed dialogue
            non_dialogue_segments = [s for s in segments if s.get("type") != "dialogue"]
            updated_segments = non_dialogue_segments + processed_dialogue
            
            # Step 4: Correct any remaining inconsistencies
            status.update(label="Correcting inconsistencies...")
            corrected_segments = self.correction_agent.correct_inconsistencies(updated_segments, entities)
            
            status.update(label="Processing complete!", state="complete")
            
        return {
            "segments": corrected_segments,
            "entities": entities
        }
    
    def generate_summary(self, result: Dict) -> Dict:
        """Generate a summary of the screenplay."""
        segments = result.get("segments", [])
        entities = result.get("entities", {})
        
        # Count segment types
        segment_types = {}
        for segment in segments:
            segment_type = segment.get("type", "unknown")
            segment_types[segment_type] = segment_types.get(segment_type, 0) + 1
        
        # Count dialogue by character
        character_dialogue = {}
        for segment in segments:
            if segment.get("type") == "dialogue":
                character = segment.get("character")
                if character:
                    character_dialogue[character] = character_dialogue.get(character, 0) + 1
        
        # Identify scene changes
        scenes = [s for s in segments if s.get("type") == "scene_header"]
        
        return {
            "segment_counts": segment_types,
            "character_dialogue_counts": character_dialogue,
            "scene_count": len(scenes),
            "character_count": len(entities.get("characters", [])),
            "location_count": len(entities.get("locations", []))
        }
        
    def export_json(self, result: Dict) -> str:
        """Export the screenplay analysis to JSON."""
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def export_csv(self, result: Dict) -> Dict[str, pd.DataFrame]:
        """Export the screenplay analysis to CSV dataframes."""
        segments = result.get("segments", [])
        
        # Convert segments to dataframes based on type
        dataframes = {}
        
        # Dialogue dataframe
        dialogue_segments = [s for s in segments if s.get("type") == "dialogue"]
        if dialogue_segments:
            dialogue_df = pd.DataFrame(dialogue_segments)
            dataframes["dialogue"] = dialogue_df
            
        # Scene header dataframe
        scene_segments = [s for s in segments if s.get("type") == "scene_header"]
        if scene_segments:
            scene_df = pd.DataFrame(scene_segments)
            dataframes["scenes"] = scene_df
            
        # Title dataframe
        title_segments = [s for s in segments if s.get("type") == "title"]
        if title_segments:
            title_df = pd.DataFrame(title_segments)
            dataframes["titles"] = title_df
            
        return dataframes

# Main Streamlit app
st.title("🎬 Screenplay Parser with LLM Agents")
st.write("Upload a screenplay document to parse and analyze it using LLM agents.")

# File upload
uploaded_file = st.file_uploader("Choose a screenplay file", type=["txt", "docx"])

if uploaded_file:
    # Read the file
    text = read_file(uploaded_file)
    
    # Show text preview
    with st.expander("Preview Text"):
        st.text_area("First 1000 characters", text[:1000], height=200)
        st.write(f"Total length: {len(text)} characters")
    
    # Process button
    if st.button("Process Screenplay"):
        # Validate configuration
        if llm_provider == "OpenAI" and not api_key:
            st.error("Please enter your OpenAI API key.")
 
        else:
            # Initialize processor with selected provider
            processor = ScreenplayProcessor(
                provider=llm_provider,
                model=model,
                api_key=api_key if llm_provider == "OpenAI" else None,
                ollama_url=ollama_url if llm_provider in ["Ollama", "DeepSeek"] else None
            )
            
            # Process the screenplay
            start_time = time.time()
            result = processor.process_screenplay(text, chunk_size=parsing_granularity)
            end_time = time.time()
            
            st.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
            
            # Display the results in tabs
            tabs = st.tabs(["Summary", "Characters", "Scenes", "Dialogue", "Raw Data"])
            
            with tabs[0]:  # Summary
                summary = processor.generate_summary(result)
                
                # Display basic stats
                st.subheader("Screenplay Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Scenes", summary["scene_count"])
                col2.metric("Characters", summary["character_count"])
                col3.metric("Locations", summary["location_count"])
                
                # Segment type distribution
                st.subheader("Segment Types")
                segment_df = pd.DataFrame({
                    "Type": list(summary["segment_counts"].keys()),
                    "Count": list(summary["segment_counts"].values())
                })
                st.bar_chart(segment_df.set_index("Type"))
                
                # Character dialogue distribution
                st.subheader("Character Dialogue Distribution")
                if summary["character_dialogue_counts"]:
                    # Sort by count
                    sorted_chars = dict(sorted(
                        summary["character_dialogue_counts"].items(),
                        key=lambda item: item[1],
                        reverse=True
                    ))
                    
                    # Display top 10
                    top_chars = {k: sorted_chars[k] for k in list(sorted_chars.keys())[:10]}
                    char_df = pd.DataFrame({
                        "Character": list(top_chars.keys()),
                        "Lines": list(top_chars.values())
                    })
                    st.bar_chart(char_df.set_index("Character"))
                else:
                    st.write("No character dialogue found.")
            
            with tabs[1]:  # Characters
                st.subheader("Characters")
                characters = result["entities"].get("characters", [])
                if characters:
                    for character in characters:
                        st.write(f"- {character}")
                    
                    # Calculate character stats
                    char_dialogue = {}
                    for segment in result["segments"]:
                        if segment.get("type") == "dialogue":
                            character = segment.get("character")
                            if character in characters:
                                char_dialogue[character] = char_dialogue.get(character, 0) + 1
                    
                    # Display character dialogue stats
                    st.subheader("Character Dialogue Stats")
                    st.dataframe(pd.DataFrame({
                        "Character": list(char_dialogue.keys()),
                        "Line Count": list(char_dialogue.values())
                    }).sort_values("Line Count", ascending=False))
                else:
                    st.write("No characters identified.")
            
            with tabs[2]:  # Scenes
                st.subheader("Scene Breakdown")
                scene_segments = [s for s in result["segments"] if s.get("type") == "scene_header"]
                
                if scene_segments:
                    # Create a scene list
                    scene_list = []
                    for i, scene in enumerate(scene_segments):
                        scene_list.append({
                            "Scene #": i + 1,
                            "Type": scene.get("scene_type", ""),
                            "Timecode": scene.get("timecode", ""),
                            "Description": scene.get("text", "")
                        })
                    
                    st.dataframe(pd.DataFrame(scene_list))
                else:
                    st.write("No scene headers identified.")
            
            with tabs[3]:  # Dialogue
                st.subheader("Dialogue Analysis")
                dialogue_segments = [s for s in result["segments"] if s.get("type") == "dialogue"]
                
                if dialogue_segments:
                    # Display some sample dialogue
                    st.write("Sample dialogue:")
                    for i, dialogue in enumerate(dialogue_segments[:10]):
                        with st.container():
                            st.write(f"**{dialogue.get('character', 'Unknown')}** ({dialogue.get('audio_type', '')})")
                            st.write(dialogue.get("text", ""))
                            st.divider()
                    
                    # Audio notation explanation
                    st.subheader("Audio Notation Guide")
                    audio_notations = result["entities"].get("audio_notations", {})
                    if audio_notations:
                        for notation, meaning in audio_notations.items():
                            st.write(f"**{notation}**: {meaning}")
                    else:
                        st.write("No audio notations identified.")
                else:
                    st.write("No dialogue identified.")
            
            with tabs[4]:  # Raw Data
                st.subheader("Raw Parsed Data")
                
                # Enable download of JSON
                st.download_button(
                    "Download JSON",
                    processor.export_json(result),
                    file_name=f"screenplay_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Show raw JSON if debug mode is enabled
                if debug_mode:
                    st.json(result)
                
                # Export to CSV
                st.subheader("Export to CSV")
                dataframes = processor.export_csv(result)
                
                for df_name, df in dataframes.items():
                    csv = df.to_csv(index=False)
                    st.download_button(
                        f"Download {df_name.capitalize()} CSV",
                        csv,
                        file_name=f"screenplay_{df_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# Show instructions when no file is uploaded
else:
    st.info("👆 Upload a screenplay file (.txt or .docx) to get started.")
    
    with st.expander("About this app"):
        st.markdown("""
        ### Screenplay Parser with LLM Agents
        
        This app uses advanced LLM (Large Language Model) agents to intelligently parse and analyze screenplay documents.
        Unlike traditional parsers that rely on fixed patterns, this app can handle:
        
        - Inconsistent formatting across different documents
        - Typos and variations in character names
        - Different audio notation styles (MO, VO, zMO, etc.)
        - Non-standard scene headers and transitions
        
        #### How it works
        
        1. **Document Segmentation**: Breaks the screenplay into logical parts
        2. **Entity Recognition**: Identifies characters, locations, and audio notations
        3. **Dialogue Processing**: Normalizes dialogue and speaker information
        4. **Correction**: Fixes inconsistencies across the document
        
        #### LLM Providers
        
        You can choose between OpenAI's API (requires API key) or a local Ollama instance
        for processing. For high accuracy, GPT-4 is recommended, but GPT-3.5-Turbo works well too.
        
        For local processing, ensure you have Ollama installed and running with your desired model.
        """)

# Footer
st.write("---")
st.caption("Screenplay Parser with LLM Agents | Created with Streamlit")