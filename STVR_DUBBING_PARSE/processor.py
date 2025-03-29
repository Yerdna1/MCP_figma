"""
Main processor for screenplay analysis that orchestrates the specialized agents
"""
import json
import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import all agents
from correction_agent import CorrectionAgent
from segmentation_agent import DocumentSegmentationAgent
from entity_agent import EntityRecognitionAgent
from dialogue_agent import DialogueProcessingAgent
from docx_export_agent import DocxExportAgent

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
        
        # Initialize the DOCX export agent (doesn't require LLM parameters)
        self.docx_export_agent = DocxExportAgent()
        
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
            
            # Log the segment count after processing
            segment_markers = len([s for s in corrected_segments if s.get("type") == "segment_marker" or 
                                  (s.get("timecode") and self._is_segment_marker(s.get("timecode")))])
            st.write(f"Found {segment_markers} segment markers in the screenplay")
            
            status.update(label="Processing complete!", state="complete")
            
        return {
            "segments": corrected_segments,
            "entities": entities
        }
    
    def _is_segment_marker(self, timecode: str) -> bool:
        """Helper method to check if a timecode represents a segment marker."""
        import re
        return bool(re.search(r'\*?\*?[\d:]+[-]{5,}', timecode) or 
                   (timecode and len(timecode.strip()) >= 4 and '-' in timecode and ':' in timecode))
    
    def export_to_docx(self, result: Dict, output_path: Optional[str] = None, episode_number: Optional[str] = None) -> str:
        """Export the processed screenplay to a formatted DOCX file."""
        # If no output path provided, create one in the temp directory
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"screenplay_formatted_{timestamp}.docx"
        
        # Get the segments from the result
        segments = result.get("segments", [])
        
        # Check if we found any segments
        if not segments:
            st.error("No segments found to export.")
            return ""
        
        # Use the DOCX export agent to create the document
        docx_path = self.docx_export_agent.export_to_docx(segments, output_path, episode_number)
        
        if not docx_path:
            st.error("Failed to create DOCX file.")
            return ""
            
        st.success(f"Successfully exported to DOCX: {docx_path}")
        return docx_path
    
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
        
        # Count segment markers
        segment_markers = len([s for s in segments if s.get("type") == "segment_marker" or 
                              (s.get("timecode") and self._is_segment_marker(s.get("timecode")))])
        
        return {
            "segment_counts": segment_types,
            "character_dialogue_counts": character_dialogue,
            "scene_count": len(scenes),
            "character_count": len(entities.get("characters", [])),
            "location_count": len(entities.get("locations", [])),
            "segment_marker_count": segment_markers
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
            
        # Segment markers dataframe
        segment_markers = [s for s in segments if s.get("type") == "segment_marker" or 
                          (s.get("timecode") and self._is_segment_marker(s.get("timecode")))]
        if segment_markers:
            markers_df = pd.DataFrame(segment_markers)
            dataframes["segments"] = markers_df
            
        return dataframes