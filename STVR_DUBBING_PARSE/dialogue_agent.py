"""
Dialogue Processing Agent for screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any

from llm_agent import LLMAgent

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