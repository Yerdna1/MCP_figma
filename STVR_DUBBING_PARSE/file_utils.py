"""
File handling utilities for the Screenplay Parser App
"""
import os
import tempfile
import docx2txt
import streamlit as st

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