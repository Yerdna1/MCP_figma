"""
Entity Recognition Agent for screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from llm_agent import LLMAgent

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