"""
Screenplay Parser App with LLM Agents
This app uses LLM agents to intelligently parse screenplay documents with inconsistent formatting.
"""
import streamlit as st
import os
import json
import time
import pandas as pd
import tempfile
from datetime import datetime


from config import setup_sidebar_config
from file_utils import read_file


# Configure page
st.title("🎬 Screenplay Parser with LLM Agents")
st.write("Upload a screenplay document to parse and analyze it using LLM agents.")

# Setup sidebar configuration
config = setup_sidebar_config()

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
        if config["llm_provider"] == "OpenAI" and not config["api_key"]:
            st.error("Please enter your OpenAI API key.")
        else:
            # Initialize processor with selected provider
            processor = ScreenplayProcessor(
                provider=config["llm_provider"],
                model=config["model"],
                api_key=config["api_key"] if config["llm_provider"] == "OpenAI" else None,
                ollama_url=config["ollama_url"] if config["llm_provider"] in ["Ollama", "DeepSeek"] else None
            )
            
            # Process the screenplay
            start_time = time.time()
            result = processor.process_screenplay(text, chunk_size=config["parsing_granularity"])
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
                if config["debug_mode"]:
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