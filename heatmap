#!/usr/bin/env python3
"""
Streamlit App for Publication-Quality Heatmaps
Upload CSV files and generate customizable heatmaps
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Page config
st.set_page_config(page_title="Heatmap Generator", layout="wide", page_icon="🔥")

# Title
st.title("🔥 Publication-Quality Heatmap Generator")
st.markdown("Upload your CSV file and generate customizable heatmaps")

# Sidebar for parameters
st.sidebar.header("Heatmap Parameters")

# Visual parameters
FIGURE_DPI = st.sidebar.slider("Figure DPI", 100, 600, 300, 50)
FIGURE_WIDTH = st.sidebar.slider("Figure Width (inches)", 4, 20, 10, 1)
FIGURE_HEIGHT = st.sidebar.slider("Figure Height (inches)", 4, 20, 8, 1)
FONT_SIZE_TITLE = st.sidebar.slider("Title Font Size", 10, 24, 14, 1)
FONT_SIZE_LABEL = st.sidebar.slider("Axis Label Font Size", 8, 20, 12, 1)
FONT_SIZE_TICK = st.sidebar.slider("Tick Font Size", 6, 16, 10, 1)

# Colormap selection
CMAP = st.sidebar.selectbox(
    "Color Map",
    ["RdBu_r", "viridis", "plasma", "coolwarm", "seismic", "bwr", "RdYlGn", "RdYlBu"]
)

# Annotation options
show_values = st.sidebar.checkbox("Show Values in Cells", value=False)
show_stars = st.sidebar.checkbox("Show Significance Stars", value=True)
ALPHA = st.sidebar.slider("Significance Threshold (α)", 0.001, 0.1, 0.05, 0.001)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("📊 Data Preview (first 10 rows)"):
            st.dataframe(df.head(10))
        
        # Column selection
        st.subheader("📋 Configure Heatmap Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            row_col = st.selectbox("Row Variable", options=df.columns.tolist())
        
        with col2:
            col_col = st.selectbox("Column Variable", options=df.columns.tolist())
        
        with col3:
            value_col = st.selectbox("Value Variable", options=df.columns.tolist())
        
        # Optional: P-value column for significance stars
        pvalue_col = None
        if show_stars:
            pvalue_col = st.selectbox(
                "P-value/FDR Column (for significance stars)",
                options=["None"] + df.columns.tolist()
            )
            if pvalue_col == "None":
                pvalue_col = None
        
        # Generate heatmap button
        if st.button("🎨 Generate Heatmap", type="primary"):
            with st.spinner("Generating heatmap..."):
                try:
                    # Create pivot table
                    pivot_data = df.pivot_table(
                        index=row_col,
                        columns=col_col,
                        values=value_col,
                        aggfunc='mean'
                    )
                    
                    # Optional: pivot for p-values
                    pivot_pvals = None
                    if pvalue_col:
                        pivot_pvals = df.pivot_table(
                            index=row_col,
                            columns=col_col,
                            values=pvalue_col,
                            aggfunc='min'
                        )
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
                    
                    # Determine color scale
                    data = pivot_data.values
                    finite_mask = np.isfinite(data)
                    vmax = np.nanmax(np.abs(data[finite_mask])) if finite_mask.any() else 1.0
                    if not np.isfinite(vmax) or vmax == 0:
                        vmax = 1.0
                    
                    # Draw heatmap
                    sns.heatmap(
                        pivot_data,
                        cmap=CMAP,
                        center=0,
                        vmin=-vmax,
                        vmax=vmax,
                        ax=ax,
                        cbar_kws={'label': value_col},
                        linewidths=0.5,
                        linecolor='gray',
                        square=False,
                        annot=show_values,
                        fmt='.2f' if show_values else ''
                    )
                    
                    # Add significance stars
                    if show_stars and pivot_pvals is not None:
                        for i in range(pivot_data.shape[0]):
                            for j in range(pivot_data.shape[1]):
                                p = pivot_pvals.iloc[i, j]
                                if pd.notna(p):
                                    stars = ""
                                    if p < 0.001:
                                        stars = "***"
                                    elif p < 0.01:
                                        stars = "**"
                                    elif p < ALPHA:
                                        stars = "*"
                                    
                                    if stars:
                                        # Determine text color based on background
                                        bg = data[i, j]
                                        norm = (bg + vmax) / (2 * vmax) if vmax > 0 else 0.5
                                        cmap_obj = plt.get_cmap(CMAP)
                                        rgba = cmap_obj(min(max(norm, 0.0), 1.0))
                                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                                        star_color = "white" if lum < 0.5 else "black"
                                        
                                        ax.text(j + 0.5, i + 0.5, stars,
                                               ha="center", va="center",
                                               fontsize=10, fontweight="bold",
                                               color=star_color)
                    
                    # Styling
                    ax.set_xlabel(col_col, fontsize=FONT_SIZE_LABEL, fontweight='bold')
                    ax.set_ylabel(row_col, fontsize=FONT_SIZE_LABEL, fontweight='bold')
                    ax.set_title(f"Heatmap: {value_col}", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
                    
                    # Rotate labels
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=FONT_SIZE_TICK)
                    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZE_TICK)
                    
                    plt.tight_layout()
                    
                    # Display
                    st.pyplot(fig)
                    
                    # Download options
                    st.subheader("💾 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Save as PNG
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
                        buf_png.seek(0)
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf_png,
                            file_name="heatmap.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # Save as SVG
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                        buf_svg.seek(0)
                        st.download_button(
                            label="📥 Download SVG",
                            data=buf_svg,
                            file_name="heatmap.svg",
                            mime="image/svg+xml"
                        )
                    
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error generating heatmap: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a CSV file to get started")
    
    # Show example data format
    with st.expander("📖 Example CSV Format"):
        st.markdown("""
        Your CSV should have at least 3 columns:
        - **Row Variable**: Categories for rows (e.g., genes, signatures)
        - **Column Variable**: Categories for columns (e.g., cell types, conditions)
        - **Value Variable**: Numeric values to plot (e.g., effect size, expression)
        - **P-value Column** (optional): For significance stars
        
        Example:
        ```
        Signature,CellType,EffectSize,PValue
        Gene1,CD4,0.5,0.001
        Gene1,CD8,-0.3,0.05
        Gene2,CD4,0.8,0.0001
        ```
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Tips")
st.sidebar.markdown("""
- Use RdBu_r for diverging data (centered at 0)
- Use viridis/plasma for sequential data
- Adjust DPI for publication quality (300+)
- SVG format is best for publications
""")
