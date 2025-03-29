"""
DOCX Export Agent for screenplay formatting and export
"""
import os
import re
from typing import Dict, List, Any, Optional
import streamlit as st
from docx import Document
from docx.shared import RGBColor, Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

class DocxExportAgent:
    """Agent for exporting screenplay data to formatted DOCX files."""
    
    def __init__(self):
        """Initialize the DOCX export agent."""
        # Define colors for different elements
        self.RED_COLOR = RGBColor(255, 0, 0)  # Speaker names
        self.BLUE_COLOR = RGBColor(0, 0, 255)  # Scene headers (INT/EXT)
        self.BLACK_COLOR = RGBColor(0, 0, 0)   # Regular text
        
        # Define font and paragraph settings
        self.FONT_NAME = 'Verdana'
        self.FONT_SIZE = 13  # Points
        self.LINE_SPACING = 1.5
        
    def export_to_docx(self, segments: List[Dict], output_path: str, episode_number: Optional[str] = None) -> str:
        """
        Export screenplay segments to a formatted DOCX file.
        
        Args:
            segments: List of screenplay segments
            output_path: Path to save the DOCX file
            episode_number: Optional episode number for segment numbering
            
        Returns:
            Path to the created DOCX file
        """
        # Create a new document
        doc = Document()
        
        # Configure document settings
        style = doc.styles['Normal']
        style.font.name = self.FONT_NAME
        style.font.size = Pt(self.FONT_SIZE)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.line_spacing = self.LINE_SPACING
        
        # Set margins
        section = doc.sections[0]
        section.left_margin = Inches(0.3)
        section.right_margin = Inches(0.5)
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        
        # Create a table for the content (2 columns: speaker and content)
        table = doc.add_table(rows=0, cols=2)
        table.autofit = False
        table.allow_autofit = False
        
        # Set column widths
        table.columns[0].width = Cm(5)    # Speaker column
        table.columns[1].width = Cm(14)   # Content column
        
        # Make table borders invisible
        for row in table.rows:
            for cell in row.cells:
                cell._tc.tcPr.tcBorders.top.val = 'nil'
                cell._tc.tcPr.tcBorders.bottom.val = 'nil'
                cell._tc.tcPr.tcBorders.left.val = 'nil'
                cell._tc.tcPr.tcBorders.right.val = 'nil'
        
        # Calculate available width for separators
        available_width = self._get_available_width(section)
        separator_length = available_width - 16  # Leave some margin
        
        # Track the current segment number
        segment_count = 0
        
        # Process each segment
        for segment in segments:
            # Check if this is a segment marker
            if segment.get("type") == "segment_marker" or (
                "timecode" in segment and self._is_segment_marker(segment.get("timecode", ""))
            ):
                # Increment segment count if not already provided
                if "segment_number" in segment:
                    segment_count = segment["segment_number"]
                else:
                    segment_count += 1
                
                # Format segment number
                if episode_number:
                    segment_number = f"{int(episode_number)}{segment_count:02d}"
                else:
                    segment_number = f"{segment_count:02d}"
                
                # Add separator line (dashes)
                self._add_separator_row(table, '-' * separator_length)
                
                # Add segment number line
                timecode = segment.get("timecode", "").split('-')[0].strip('*')  # Extract the timecode part
                
                # Calculate spacing to position segment number near the right margin
                right_margin = 2  # Number of spaces from right edge
                spacing = separator_length - len(timecode) - len(segment_number) - right_margin
                
                # Format the segment text with the timecode at left, segment number at right
                segment_text = f"{timecode}" + " " * spacing + f"{segment_number}"
                
                # Add to table with bold text but no spacing after
                self._add_segment_number_row(table, segment_text)
                
                continue
            
            # Handle scene headers (INT/EXT)
            if segment.get("text", "").upper().startswith("INT") or segment.get("text", "").upper().startswith("EXT"):
                scene_text = segment.get("text", "")
                
                # Try to extract the scene type (INT/EXT)
                scene_match = re.match(r'^(INT|EXT)\.?\s*(.*)$', scene_text, re.IGNORECASE)
                if scene_match:
                    scene_type = scene_match.group(1).upper()
                    scene_desc = scene_match.group(2).strip()
                    self._add_split_content(table, scene_type, scene_desc, speaker_color=self.BLUE_COLOR)
                else:
                    self._add_split_content(table, "", scene_text)
                
                continue
            
            # Handle speakers with dialogue
            if "speaker" in segment:
                speaker = segment.get("speaker", "")
                text = segment.get("text", "")
                
                # If there's a timecode, add it before the speaker
                if "timecode" in segment and segment["timecode"] and not self._is_segment_marker(segment["timecode"]):
                    speaker = f"{segment['timecode']} {speaker}"
                
                self._add_split_content(table, speaker, text, speaker_color=self.RED_COLOR)
                continue
            
            # Handle any other content
            self._add_split_content(table, "", segment.get("text", ""))
        
        # Save the document
        try:
            doc.save(output_path)
            return output_path
        except Exception as e:
            st.error(f"Error saving DOCX file: {e}")
            return ""
    
    def _is_segment_marker(self, timecode: str) -> bool:
        """Check if a timecode is a segment marker (contains multiple dashes)."""
        return bool(re.search(r'\*?\*?[\d:]+[-]{5,}', timecode) or 
                   (timecode and len(timecode.strip()) >= 4 and '-' in timecode and ':' in timecode))
    
    def _get_available_width(self, section) -> int:
        """Calculate available width in characters based on page settings."""
        # Get page width in inches
        page_width = section.page_width.inches
        # Subtract margins
        available_width = page_width - section.left_margin.inches - section.right_margin.inches + 2.7
        # Convert to approximate character count (assuming each char is ~0.1 inches)
        char_count = int(available_width / 0.1)
        return char_count
    
    def _add_separator_row(self, table, text, is_bold=False):
        """Add a separator row with dashes."""
        row = table.add_row()
        cell = row.cells[0].merge(row.cells[1])
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(text)
        
        # Apply formatting
        run.font.name = self.FONT_NAME
        run.font.size = Pt(self.FONT_SIZE)
        run.font.bold = is_bold
        
        # Set spacing
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(6)  # Add space after separator
        paragraph.paragraph_format.line_spacing = self.LINE_SPACING
        
        return row
    
    def _add_segment_number_row(self, table, text):
        """Add a segment number row with proper formatting."""
        row = table.add_row()
        cell = row.cells[0].merge(row.cells[1])
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(text)
        
        # Apply formatting
        run.font.name = self.FONT_NAME
        run.font.size = Pt(self.FONT_SIZE)
        run.font.bold = True  # Bold for segment numbers
        
        # Set spacing - NO spacing after segment number
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)  # No space after segment number
        paragraph.paragraph_format.line_spacing = self.LINE_SPACING
        
        return row
    
    def _add_split_content(self, table, speaker, content, speaker_color=None):
        """Add a row with speaker and content in separate columns."""
        row = table.add_row()
        
        # Speaker cell
        if speaker:
            speaker_para = row.cells[0].paragraphs[0]
            speaker_run = speaker_para.add_run(speaker)
            
            # Apply formatting
            speaker_run.font.name = self.FONT_NAME
            speaker_run.font.size = Pt(self.FONT_SIZE)
            if speaker_color:
                speaker_run.font.color.rgb = speaker_color
            
            # Set spacing
            speaker_para.paragraph_format.space_before = Pt(0)
            speaker_para.paragraph_format.space_after = Pt(0)
            speaker_para.paragraph_format.line_spacing = self.LINE_SPACING
        
        # Content cell
        if content:
            content_para = row.cells[1].paragraphs[0]
            content_run = content_para.add_run(content)
            
            # Apply formatting
            content_run.font.name = self.FONT_NAME
            content_run.font.size = Pt(self.FONT_SIZE)
            
            # Set spacing
            content_para.paragraph_format.space_before = Pt(0)
            content_para.paragraph_format.space_after = Pt(0)
            content_para.paragraph_format.line_spacing = self.LINE_SPACING
        
        return row