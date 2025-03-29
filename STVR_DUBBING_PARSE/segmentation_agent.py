"""
Document Segmentation Agent for screenplay parsing
"""
import re
import json
import streamlit as st
from typing import Dict, List, Any
from llm_agent import LLMAgent  # Import LLMAgent from its module






class DocumentSegmentationAgent(LLMAgent):
    """Agent for segmenting the document into logical parts."""
    
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