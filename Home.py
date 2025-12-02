#!/usr/bin/env python3
"""
Home page for Publication Visualization Suite
Multi-page Streamlit app launcher
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Visualization Suite",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Publication-Quality Visualization Suite")
st.markdown("### Generate professional scientific visualizations from CSV files")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome! This suite provides three powerful tools for creating publication-quality visualizations:
    
    **Choose an app from the sidebar to get started →**
    """)

with col2:
    st.info("💡 **New to Streamlit?**\n\nUse the sidebar to navigate between apps!")

# App cards
st.markdown("## 🎨 Available Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔥 Heatmap Generator
    
    Create interactive heatmaps with:
    - Customizable color schemes
    - Significance stars
    - Publication-ready styling
    - Multiple export formats
    
    **Best for:** Gene expression, correlation matrices, comparison studies
    """)
    
with col2:
    st.markdown("""
    ### ⭕ Venn Diagrams
    
    Generate Venn diagrams with:
    - 2-way and 3-way support
    - Overlap statistics
    - Custom colors
    - Interactive exploration
    
    **Best for:** Set comparisons, gene lists, categorical overlaps
    """)

with col3:
    st.markdown("""
    ### 📊 Bar Graphs
    
    Build bar charts with:
    - Multiple orientations
    - Grouped/stacked options
    - Error bars
    - Statistical summaries
    
    **Best for:** Group comparisons, distributions, summary statistics
    """)

st.markdown("---")

# Quick start guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    ### Getting Started
    
    1. **Select an app** from the sidebar
    2. **Upload your CSV file** using the file uploader
    3. **Configure parameters** using the controls
    4. **Generate your visualization**
    5. **Download** in PNG or SVG format
    
    ### CSV Format Tips
    
    - Use comma-separated values
    - Include column headers
    - Remove special characters
    - Check for missing values
    
    ### Export Options
    
    - **PNG**: Great for presentations and web
    - **SVG**: Best for publications (vector format)
    - **High DPI**: Use 300+ for print quality
    """)

# Features
with st.expander("✨ Key Features"):
    st.markdown("""
    ### Publication Quality
    - High-resolution output (up to 600 DPI)
    - Vector graphics support (SVG)
    - Customizable fonts and colors
    - Professional styling
    
    ### User Friendly
    - No coding required
    - Interactive controls
    - Real-time preview
    - Data validation
    
    ### Flexible
    - Multiple visualization types
    - Various export formats
    - Customizable parameters
    - Batch processing support
    """)

# Example data
with st.expander("📖 Example Data Formats"):
    st.markdown("""
    ### Heatmap Data
    ```csv
    Signature,CellType,EffectSize,PValue
    Gene1,CD4,0.5,0.001
    Gene1,CD8,-0.3,0.05
    ```
    
    ### Venn Diagram Data
    ```csv
    Set1,Set2,Set3
    ItemA,ItemB,ItemA
    ItemB,ItemC,ItemD
    ```
    
    ### Bar Graph Data
    ```csv
    Category,Value,Error
    GroupA,10.5,1.2
    GroupB,15.2,0.8
    ```
    """)

st.markdown("---")

# Footer
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📚 Documentation
    - Check the README for detailed instructions
    - Each app has built-in examples
    - Hover over controls for tooltips
    """)

with col2:
    st.markdown("""
    ### 💾 Output Formats
    - PNG (raster images)
    - SVG (vector graphics)
    - CSV (processed data)
    """)

with col3:
    st.markdown("""
    ### 🎯 Best Practices
    - Use 300+ DPI for publications
    - Export SVG for journals
    - Preview before downloading
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Ready for scientific publication 📄</p>
</div>
""", unsafe_allow_html=True)
