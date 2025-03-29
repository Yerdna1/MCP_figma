"""
Correction Agent for fixing inconsistencies in screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from llm_agent import LLMAgent

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