"""
Base LLM Agent class for the Screenplay Parser App
"""
import re
import json
import time
import requests
import streamlit as st
from typing import Optional, Dict, List, Any

from config import MAX_RETRIES, RETRY_DELAY, TEMPERATURE

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
                    if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
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
            
            if hasattr(st, 'session_state') and st.session_state.get('detailed_progress', True):
                st.write(f"Calling DeepSeek model: {self.model}")
            
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
                st.write(f"DeepSeek prompt length: {len(full_prompt)} characters")
            
            # Make the API call with configured timeout
            timeout_seconds = st.session_state.get('timeout_seconds', 120)
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
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
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
            st.warning(f"DeepSeek request timed out after {st.session_state.get('timeout_seconds', 120)} seconds.")
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
            if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
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
                if hasattr(st, 'session_state') and st.session_state.get('debug_mode', False):
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