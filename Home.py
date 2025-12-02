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

# App links configuration
APP_LINKS = {
    "Bar Graph Generator": "https://webtools-bar.streamlit.app/",
    "Heatmap Generator": "https://webtools-heatmap.streamlit.app/",
    "Venn Diagram Generator": "https://webtools-venn.streamlit.app/"
}

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome! This suite provides three powerful tools for creating publication-quality visualizations:
    
    **Click on the buttons below to launch each visualization tool:**
    """)

with col2:
    st.info("💡 **New to Streamlit?**\n\nClick the buttons below to launch each tool in a new tab!")

# App cards with links
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
    
    # Link button for Heatmap Generator
    st.link_button(
        "🚀 Launch Heatmap Generator",
        APP_LINKS["Heatmap Generator"],
        help="Opens the Heatmap Generator in a new tab",
        use_container_width=True
    )

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
    
    # Link button for Venn Diagram Generator
    st.link_button(
        "🚀 Launch Venn Diagram Generator",
        APP_LINKS["Venn Diagram Generator"],
        help="Opens the Venn Diagram Generator in a new tab",
        use_container_width=True
    )

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
    
    # Link button for Bar Graph Generator
    st.link_button(
        "🚀 Launch Bar Graph Generator",
        APP_LINKS["Bar Graph Generator"],
        help="Opens the Bar Graph Generator in a new tab",
        use_container_width=True
    )

st.markdown("---")

# Quick launch section
st.markdown("## 🚀 Quick Launch")
st.markdown("Click any link below to open the corresponding app:")

# Create quick launch buttons in a row
launch_col1, launch_col2, launch_col3 = st.columns(3)

with launch_col1:
    if st.button("📊 **Bar Graph Generator**", use_container_width=True):
        st.markdown(f"[Click here to open Bar Graph Generator ↗]({APP_LINKS['Bar Graph Generator']})")

with launch_col2:
    if st.button("🔥 **Heatmap Generator**", use_container_width=True):
        st.markdown(f"[Click here to open Heatmap Generator ↗]({APP_LINKS['Heatmap Generator']})")

with launch_col3:
    if st.button("⭕ **Venn Diagram Generator**", use_container_width=True):
        st.markdown(f"[Click here to open Venn Diagram Generator ↗]({APP_LINKS['Venn Diagram Generator']})")

# Quick start guide
with st.expander("📖 Quick Start Guide", expanded=True):
    st.markdown("""
    ### Getting Started
    
    1. **Click any of the launch buttons above** to open the corresponding app
    2. **Upload your CSV file** using the file uploader
    3. **Configure parameters** using the controls
    4. **Generate your visualization**
    5. **Download** in PNG or SVG format
    
    ### Direct App Links:
    - **[Bar Graph Generator]({bar_link})** - Create publication-quality bar charts
    - **[Heatmap Generator]({heatmap_link})** - Generate interactive heatmaps
    - **[Venn Diagram Generator]({venn_link})** - Create Venn/Euler diagrams
    
    ### CSV Format Tips
    - Use comma-separated values
    - Include column headers
    - Remove special characters
    - Check for missing values
    
    ### Export Options
    - **PNG**: Great for presentations and web
    - **SVG**: Best for publications (vector format)
    - **High DPI**: Use 300+ for print quality
    """.format(
        bar_link=APP_LINKS["Bar Graph Generator"],
        heatmap_link=APP_LINKS["Heatmap Generator"],
        venn_link=APP_LINKS["Venn Diagram Generator"]
    ))

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

# Footer with direct links
st.markdown("## 🔗 Direct Links to Apps")
link_col1, link_col2, link_col3 = st.columns(3)

with link_col1:
    st.markdown(f"""
    ### 📊 Bar Graph Generator
    [Open Bar Graph Generator ↗]({APP_LINKS['Bar Graph Generator']})
    
    **Features:**
    - Multiple orientations
    - Grouped/stacked options
    - Error bars
    - Statistical summaries
    """)

with link_col2:
    st.markdown(f"""
    ### 🔥 Heatmap Generator
    [Open Heatmap Generator ↗]({APP_LINKS['Heatmap Generator']})
    
    **Features:**
    - Custom color schemes
    - Significance stars
    - Publication-ready styling
    - Interactive exploration
    """)

with link_col3:
    st.markdown(f"""
    ### ⭕ Venn Diagram Generator
    [Open Venn Diagram Generator ↗]({APP_LINKS['Venn Diagram Generator']})
    
    **Features:**
    - 2-way and 3-way support
    - Overlap statistics
    - Custom colors
    - Euler diagram support
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit 🎈 | Ready for scientific publication 📄</p>
</div>
""", unsafe_allow_html=True)

# Add some JavaScript to make links open in new tab by default
st.markdown("""
<script>
// Make all external links open in new tab
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="http"]');
    links.forEach(link => {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
    });
});
</script>
""", unsafe_allow_html=True)
