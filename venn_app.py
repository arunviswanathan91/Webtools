#!/usr/bin/env python3
"""
Streamlit App for Venn Diagrams
Upload CSV files and generate 2-way or 3-way Venn diagrams
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
import io

# Page config
st.set_page_config(page_title="Venn Diagram Generator", layout="wide", page_icon="⭕")

# Title
st.title("⭕ Venn Diagram Generator")
st.markdown("Upload your CSV file and generate publication-quality Venn diagrams")

# Sidebar for parameters
st.sidebar.header("Venn Diagram Parameters")

# Visual parameters
FIGURE_DPI = st.sidebar.slider("Figure DPI", 100, 600, 300, 50)
FIGURE_WIDTH = st.sidebar.slider("Figure Width (inches)", 4, 16, 8, 1)
FIGURE_HEIGHT = st.sidebar.slider("Figure Height (inches)", 4, 16, 8, 1)
FONT_SIZE_TITLE = st.sidebar.slider("Title Font Size", 10, 24, 16, 1)
FONT_SIZE_LABEL = st.sidebar.slider("Set Label Font Size", 8, 20, 14, 1)

# Color options
st.sidebar.subheader("Colors")
color1 = st.sidebar.color_picker("Set 1 Color", "#FF6B6B")
color2 = st.sidebar.color_picker("Set 2 Color", "#4ECDC4")
color3 = st.sidebar.color_picker("Set 3 Color", "#45B7D1")

# Alpha/transparency
alpha = st.sidebar.slider("Transparency", 0.0, 1.0, 0.5, 0.1)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("📊 Data Preview (first 10 rows)"):
            st.dataframe(df.head(10))
        
        # Method selection
        st.subheader("📋 Configure Venn Diagram")
        
        method = st.radio(
            "Data Format",
            ["Separate Columns (each column is a set)", "Set Name & Item Columns"]
        )
        
        if method == "Separate Columns (each column is a set)":
            st.markdown("""
            **Expected Format**: Each column contains items (one per row). 
            Empty cells are ignored. Up to 3 sets supported.
            """)
            
            # Select columns for sets
            available_cols = df.columns.tolist()
            
            num_sets = st.radio("Number of Sets", [2, 3])
            
            if num_sets == 2:
                col1, col2 = st.columns(2)
                with col1:
                    set1_col = st.selectbox("Set 1 Column", available_cols, key='set1')
                with col2:
                    set2_col = st.selectbox("Set 2 Column", available_cols, key='set2', index=min(1, len(available_cols)-1))
                
                set1_label = st.text_input("Set 1 Label", value=set1_col)
                set2_label = st.text_input("Set 2 Label", value=set2_col)
                
            else:  # 3 sets
                col1, col2, col3 = st.columns(3)
                with col1:
                    set1_col = st.selectbox("Set 1 Column", available_cols, key='set1')
                with col2:
                    set2_col = st.selectbox("Set 2 Column", available_cols, key='set2', index=min(1, len(available_cols)-1))
                with col3:
                    set3_col = st.selectbox("Set 3 Column", available_cols, key='set3', index=min(2, len(available_cols)-1))
                
                set1_label = st.text_input("Set 1 Label", value=set1_col)
                set2_label = st.text_input("Set 2 Label", value=set2_col)
                set3_label = st.text_input("Set 3 Label", value=set3_col)
            
            if st.button("🎨 Generate Venn Diagram", type="primary"):
                with st.spinner("Generating Venn diagram..."):
                    try:
                        # Extract sets (remove NaN values)
                        set1 = set(df[set1_col].dropna().astype(str).unique())
                        set2 = set(df[set2_col].dropna().astype(str).unique())
                        
                        if num_sets == 3:
                            set3 = set(df[set3_col].dropna().astype(str).unique())
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
                        
                        if num_sets == 2:
                            # 2-way Venn
                            venn = venn2(
                                [set1, set2],
                                set_labels=(set1_label, set2_label),
                                set_colors=(color1, color2),
                                alpha=alpha,
                                ax=ax
                            )
                            venn2_circles([set1, set2], ax=ax, linewidth=2)
                        else:
                            # 3-way Venn
                            venn = venn3(
                                [set1, set2, set3],
                                set_labels=(set1_label, set2_label, set3_label),
                                set_colors=(color1, color2, color3),
                                alpha=alpha,
                                ax=ax
                            )
                            venn3_circles([set1, set2, set3], ax=ax, linewidth=2)
                        
                        # Style labels
                        for text in venn.set_labels:
                            if text:
                                text.set_fontsize(FONT_SIZE_LABEL)
                                text.set_fontweight('bold')
                        
                        for text in venn.subset_labels:
                            if text:
                                text.set_fontsize(FONT_SIZE_LABEL - 2)
                        
                        plt.title("Venn Diagram", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
                        plt.tight_layout()
                        
                        # Display
                        st.pyplot(fig)
                        
                        # Show statistics
                        st.subheader("📊 Set Statistics")
                        if num_sets == 2:
                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"{set1_label} only", len(set1 - set2))
                            col2.metric("Intersection", len(set1 & set2))
                            col3.metric(f"{set2_label} only", len(set2 - set1))
                            
                            # Show actual overlapping items
                            with st.expander("🔍 View Overlapping Items"):
                                overlap = list(set1 & set2)
                                if overlap:
                                    st.write(f"**{len(overlap)} items in common:**")
                                    st.write(overlap)
                                else:
                                    st.write("No overlapping items")
                        
                        else:  # 3 sets
                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"{set1_label} Total", len(set1))
                            col2.metric(f"{set2_label} Total", len(set2))
                            col3.metric(f"{set3_label} Total", len(set3))
                            
                            col1, col2 = st.columns(2)
                            col1.metric("All 3 Sets", len(set1 & set2 & set3))
                            col2.metric("Any 2 Sets", len((set1 & set2) | (set1 & set3) | (set2 & set3)) - len(set1 & set2 & set3))
                            
                            with st.expander("🔍 View Overlapping Items"):
                                st.write("**All 3 sets:**", list(set1 & set2 & set3))
                                st.write(f"**{set1_label} & {set2_label} only:**", list((set1 & set2) - set3))
                                st.write(f"**{set1_label} & {set3_label} only:**", list((set1 & set3) - set2))
                                st.write(f"**{set2_label} & {set3_label} only:**", list((set2 & set3) - set1))
                        
                        # Download options
                        st.subheader("💾 Download Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            buf_png = io.BytesIO()
                            fig.savefig(buf_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
                            buf_png.seek(0)
                            st.download_button(
                                label="📥 Download PNG",
                                data=buf_png,
                                file_name="venn_diagram.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            buf_svg = io.BytesIO()
                            fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                            buf_svg.seek(0)
                            st.download_button(
                                label="📥 Download SVG",
                                data=buf_svg,
                                file_name="venn_diagram.svg",
                                mime="image/svg+xml"
                            )
                        
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating Venn diagram: {str(e)}")
                        st.exception(e)
        
        else:  # Set Name & Item format
            st.markdown("""
            **Expected Format**: 
            - One column with set/category names
            - One column with item names
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                set_col = st.selectbox("Set/Category Column", df.columns.tolist())
            with col2:
                item_col = st.selectbox("Item Column", df.columns.tolist())
            
            # Show unique sets
            unique_sets = df[set_col].unique()
            st.info(f"Found {len(unique_sets)} unique sets: {', '.join(map(str, unique_sets[:5]))}{'...' if len(unique_sets) > 5 else ''}")
            
            if len(unique_sets) < 2 or len(unique_sets) > 3:
                st.warning("⚠️ Please select columns that result in 2-3 unique sets")
            else:
                if st.button("🎨 Generate Venn Diagram", type="primary"):
                    with st.spinner("Generating Venn diagram..."):
                        try:
                            # Group items by set
                            grouped = df.groupby(set_col)[item_col].apply(lambda x: set(x.dropna().astype(str)))
                            sets_dict = grouped.to_dict()
                            
                            set_names = list(sets_dict.keys())
                            set_values = list(sets_dict.values())
                            
                            # Create figure
                            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
                            
                            if len(set_names) == 2:
                                venn = venn2(
                                    set_values,
                                    set_labels=set_names,
                                    set_colors=(color1, color2),
                                    alpha=alpha,
                                    ax=ax
                                )
                                venn2_circles(set_values, ax=ax, linewidth=2)
                            else:
                                venn = venn3(
                                    set_values,
                                    set_labels=set_names,
                                    set_colors=(color1, color2, color3),
                                    alpha=alpha,
                                    ax=ax
                                )
                                venn3_circles(set_values, ax=ax, linewidth=2)
                            
                            for text in venn.set_labels:
                                if text:
                                    text.set_fontsize(FONT_SIZE_LABEL)
                                    text.set_fontweight('bold')
                            
                            for text in venn.subset_labels:
                                if text:
                                    text.set_fontsize(FONT_SIZE_LABEL - 2)
                            
                            plt.title("Venn Diagram", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Download
                            col1, col2 = st.columns(2)
                            with col1:
                                buf_png = io.BytesIO()
                                fig.savefig(buf_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
                                buf_png.seek(0)
                                st.download_button("📥 Download PNG", buf_png, "venn_diagram.png", "image/png")
                            with col2:
                                buf_svg = io.BytesIO()
                                fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                                buf_svg.seek(0)
                                st.download_button("📥 Download SVG", buf_svg, "venn_diagram.svg", "image/svg+xml")
                            
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to get started")
    
    with st.expander("📖 Example CSV Formats"):
        st.markdown("""
        **Method 1: Separate Columns**
        ```
        Set1,Set2,Set3
        Gene1,Gene2,Gene1
        Gene2,Gene3,Gene4
        Gene3,Gene4,Gene5
        ```
        
        **Method 2: Set Name & Item**
        ```
        SetName,Item
        Upregulated,Gene1
        Upregulated,Gene2
        Downregulated,Gene3
        Downregulated,Gene1
        ```
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Tips")
st.sidebar.markdown("""
- Supports 2-way and 3-way Venn diagrams
- Adjust transparency to see overlaps better
- Use contrasting colors for clarity
- Export SVG for publications
""")
